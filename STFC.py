# -*- coding: utf-8 -*-

import boto3
import time
import re
import sys
import io
import networkx as nx
import pandas as pd
import collections
import numpy as np

## Set your AWS Athena database information
## The following information is not valid
athena_s3_bucket = "company_s3_bucket_name"
aws_region = "company_s3_region"

## Set your access and secret key values here
## the following is not valid
aws_access_key_id = "ABCDEFG12345"
aws_secret_access_key = "abcdefg!@#$%"


#====================================================
# S3 Methods wrapped in class
#====================================================
class S3:
    def __init__(self, aws_access_key_id, aws_secret_access_key, aws_region=aws_region):
        self.s3Client = boto3.client('s3', region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key)

        self.s3Res = boto3.resource('s3', region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key)

    #====================================================
    # Get object from s3
    #====================================================
    def s3_get_object(self, SrcBucket, SrcKey):
        fileobj = self.s3Client.get_object(Bucket=SrcBucket, Key=SrcKey)
        return fileobj

#====================================================
# Athena Methods wrapped in class
#====================================================
class Athena:
    def __init__(self, aws_access_key_id, aws_secret_access_key, aws_region=aws_region):
        self.athenaClient = boto3.client('athena', region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key)

    #====================================================
    # Poll Athena Query Status
    #====================================================
    def poll_query_status(self, _id):
        state  = 'RUNNING'
        while state == 'RUNNING' or state == 'QUEUED':
            result = self.athenaClient.get_query_execution( QueryExecutionId = _id )
            if 'QueryExecution' in result and 'Status' in result['QueryExecution'] and 'State' in result['QueryExecution']['Status']:
                state = result['QueryExecution']['Status']['State']
                if state == 'FAILED':
                    return result
                elif state == 'SUCCEEDED':
                    return result
                elif state == 'RUNNING':
                    time.sleep(2)
                elif state == 'QUEUED':
                    time.sleep(1)
                else:
                    raise Exception(f"Encountered unknown state in athena poll query: {state}")

    #====================================================
    # Run Athena Query with Polling
    #====================================================
    def run_athena_query(self, query, log=False, s3_output=athena_s3_bucket, response_req=False):
        try:
            if log:
                print("-----------------------------------------------------------------------")
                print(query)
            response = self.athenaClient.start_query_execution(
                QueryString=query,
                ResultConfiguration={
                    'OutputLocation': "s3://"+s3_output},
                WorkGroup="users_quality")###############################change this part

            QueryExecutionId = response['QueryExecutionId']
            result = self.poll_query_status(QueryExecutionId)

            if result['QueryExecution']['Status']['State'] == 'SUCCEEDED':
                if log:
                    print("Query SUCCEEDED: {}".format(QueryExecutionId))
                    print("Time Taken (millisec) : " + str(result['QueryExecution']['Statistics']['EngineExecutionTimeInMillis']))
                    print("Data Scanned (bytes)  : " + str(result['QueryExecution']['Statistics']['DataScannedInBytes']))
                    print("-----------------------------------------------------------------------")
                if response_req:
                    s3_path = result['QueryExecution']['ResultConfiguration']['OutputLocation']
                    filename = re.findall(r'.*\/(.*)', s3_path)[0]
                    return filename
                else:
                    return True
            else:
                print(result['QueryExecution']['Status']['StateChangeReason'])
                if log:
                    print("Query " + result['QueryExecution']['Status']['State'] + ": {}".format(QueryExecutionId))
                    print("Time Taken (millisec) : " + str(result['QueryExecution']['Statistics']['EngineExecutionTimeInMillis']))
                    print("Data Scanned (bytes)  : " + str(result['QueryExecution']['Statistics']['DataScannedInBytes']))
                    print("-----------------------------------------------------------------------")
                return False
        except Exception as e:
            print("SQL Query execution failed. Check logs for more details")
            exc_type, _, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno, e)
            return False
    
#STEP 1: loading data
##getting backlog data
print('Loading Data - Start')
athena_obj = Athena(aws_access_key_id, aws_secret_access_key)
s3_obj = S3(aws_access_key_id, aws_secret_access_key)

## Set the query you want to fire
athena_query_backlog = """SELECT sales_order_num, line, flow_status_code, line_type, integration_id, split_from_line_id
from schema.oracle_shipment_backlog_table;"""

athena_output_filename = athena_obj.run_athena_query(athena_query_backlog, log=False, response_req=True)
if athena_output_filename:
    readobj = s3_obj.s3_get_object(athena_s3_bucket, athena_output_filename)
    df_backlog = pd.read_csv(io.BytesIO(readobj['Body'].read()), encoding='utf8')
    print(df_backlog.shape)
else:
    print("Backlog Query execution failed")
    
athena_query_history = """SELECT *
from schema.oracle_order_history_table;"""

##getting history data
athena_output_filename = athena_obj.run_athena_query(athena_query_history, log=False, response_req=True)
if athena_output_filename:
    readobj = s3_obj.s3_get_object(athena_s3_bucket, athena_output_filename)
    df_history = pd.read_csv(io.BytesIO(readobj['Body'].read()), encoding='utf8')
    print(df_history.shape)
else:
    print("History Query execution failed")
print('Loading Data - Finish')
    
#STEP 2: basic filtering on history
#line filtering with only forward values
print('Data Cleansing on History - Start')
df=df_history
df_temp=df[df['forward_reverse_flag']=="F"]
#line type filtering
line_type=['standard ship line', 'vmi replenishment intraco line', 'vmi replenish interco line', 'blanket']
df_temp=df_temp[df_temp['order_line_type'].str.contains('|'.join(line_type), case=False, na=False, regex=True)] 
#remove canceled lines
df_temp=df_temp[~(df_temp['line_status_code'].str.lower()=="cancelled")]
print('Data Cleansing - Filtering on history - Finish')

#STEP 3: new essentail columns on History
#original booking
df_temp['original_booking_flag']=df.apply(lambda x: 'original' if str(x.line)[-2:]==".1" else 'split', axis=1)
df_temp['SO_LN']=df.apply(lambda x: str(x.sales_order_number)+'_'+str(x.line), axis=1)
df=df_temp
print('Data Cleansing - New Fields on history - Finish')
print('Data Cleansing on History - Finish') 

df['Churn'] = df['Churn'].mask(df['sales_trans_group_type_desc'].ne('Schedule Date Change'))
df['CRD_Change'] = df['CRD_Change'].mask(df['sales_trans_group_type_desc'].ne('Requested Date Change'))
df_Churn=df.groupby(['SO_LN'])[['Churn','CRD_Change']].count()#.reset_index(name=['Churn','CRD_Change'])
df_Churn.reset_index(inplace=True)
df.drop(['Churn', 'CRD_Change'], axis=1, inplace=True)

#STEP 4-1: Gorup one with Customer request
print('SOLN with Customer Request - Start') 
df_g1=df[df['sales_trans_group_type_desc']=='Requested Date Change']
idx1=df_g1.groupby(['SO_LN'])['last_update_date'].transform(max) == df_g1['last_update_date']
df_g1 = df_g1[idx1]

df_g1_SOLN=df_g1[['SO_LN', 'last_update_date']]
df_g1_SOLN.rename(columns={'last_update_date':'CRD_last_update_date'}, inplace=True)

#get PD Change rows only from the original table
df_PD=df[df['sales_trans_group_type_desc']=='Promised Date Change']
#join with SO_LN
df_PD_g1=df_PD.merge(df_g1_SOLN, how='inner', on='SO_LN')
df_PD_g1['after_CRD']=df_PD_g1.apply(lambda x: x.last_update_date>=x.CRD_last_update_date, axis=1)

df_PD_g1=df_PD_g1[df_PD_g1['after_CRD']==True]
idx1=df_PD_g1.groupby(['SO_LN'])['last_update_date'].transform(min) == df_PD_g1['last_update_date']
df_PD_g1 = df_PD_g1[idx1]
print('SOLN with Customer Request - Finish') 

#STEP 4-2: Group with PD change available
print('SOLN with First PD Change - Start') 
df_g23=df[~df.SO_LN.isin(df_PD_g1.SO_LN)]
df_g2=df_g23[df_g23['sales_trans_group_type_desc']=='Promised Date Change']

idx2=df_g2.groupby(['SO_LN'])['last_update_date'].transform(min) == df_g2['last_update_date']
df_PD_g2=df_g2[idx2]
print('SOLN with First PD Change - Finish') 

#STEP 4-3: Group with PD inthe first line
print('SOLN with PD in the first line - Start') 
df_g3=df_g23[~df_g23.SO_LN.isin(df_PD_g2.SO_LN)]
idx3=df_g3.groupby(['SO_LN'])['last_update_date'].transform(min) == df_g3['last_update_date']
df_PD_g3=df_g3[idx3]
print('SOLN with PD in the first line - Finish')

#STEP 5: Concat all the PD dfs
df_PD_g1.drop(['CRD_last_update_date','after_CRD'], axis=1, inplace=True)
df_PD_g123=pd.concat([df_PD_g1, df_PD_g2, df_PD_g3])

#STEP 6: Data cleansing on backlog
print('Data Cleansing on Backlog - Start')
df=df_backlog
df['SO_LN']=df.apply(lambda x: str(x.sales_order_num)+'_'+str(x.line), axis=1)
df=df[df['flow_status_code']!='CANCELLED']
line_type=['standard ship line', 'vmi replenishment intraco line', 'vmi replenish interco line', 'blanket']
df=df[df['line_type'].str.contains('|'.join(line_type), case=False, na=False, regex=True)] 
print('Data Cleansing on Backlog - Finish')

#STEP 7: Tracing to the first PD
print('First PD Trace Back - Start')
df=df_PD_g123.merge(df[['SO_LN', 'integration_id', 'split_from_line_id']], how='inner', on='SO_LN')
#data cleansing: to float beacuse of null & label parent nodes as 0.0
df['split_from_line_id']=df['split_from_line_id'].astype('float')
df['split_from_line_id']=df['split_from_line_id'].apply(lambda x: 0.0 if pd.isna(x) else x)
df['split_from_line_id']=df['split_from_line_id'].astype('int')

#'level' field: the number represents how many parents does a line have
G = nx.DiGraph()
G.add_nodes_from(set(df['integration_id'].unique()).union(set(df.split_from_line_id.unique())))
G.add_edges_from([(float(r[1]['split_from_line_id']), float(r[1]['integration_id'])) for r in df.iterrows()])
df['level'] = df['integration_id'].map(nx.shortest_path_length(G, 0))
print('level field Generated')

#generate dict for
dict_id = df.set_index('integration_id').split_from_line_id.to_dict()
dict_PD = df.set_index('integration_id').promise_date.to_dict()

def get_parent_id(anc):
    """
    recursive function to grab its parent's id and its ancestary
    this function is used for apply function
    """
    
    #create a list
    anc = [anc] if not isinstance(anc, list) else anc

    #if it is a original node, then stop adding path
    if anc[-1] == 0:
        return anc

    else: #add if it is not the parent node
        if dict_id[anc[-1]] in dict_id:
            parent = get_parent_id([dict_id[anc[-1]]])
        else:
            parent = [0]            
        anc += parent
        
        return anc

#'First PD' field
df['path_id']=df.integration_id.apply(get_parent_id)
df['path_PD']=df.apply(lambda x: [dict_PD[id_] for id_ in x.path_id if not (id_==x.integration_id or id_==0)], axis=1)
df['First_PD']=df.apply(lambda x: x.promise_date if x.path_PD==[] else list(x.path_PD)[-1], axis=1)
print('First PD Trace Back - Finish')

#STEP 8: 
print('Generating final out put - Start')
df_PD_FINAL=df_PD_g123.merge(df[['SO_LN','level']], how='left', on='SO_LN')
df_PD_g23=pd.concat([df_PD_g2,df_PD_g3])
df_g23_SOLN=df_PD_g23['SO_LN']
df=df[['SO_LN','First_PD']]
df_first_PD_g23=df[df.SO_LN.isin(set(df_g23_SOLN))]
df_PD_FINAL=df_PD_FINAL.merge(df_first_PD_g23, how='left', on='SO_LN')
df_PD_FINAL['First_PD']=df_PD_FINAL.apply(lambda x: x.promise_date if pd.isnull(x.First_PD) else x.First_PD, axis=1)
df_PD_FINAL=df_PD_FINAL.merge(df_Churn, how='left', on='SO_LN')
print(df_PD_FINAL.shape)

#df_PD_FINAL output save
df_PD_FINAL.to_csv(r'C:\file_location\df_PD_FINAL_example.csv', index=False)
print('Generating final out put - End')

