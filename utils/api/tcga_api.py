import os
import io
import re
import json
import tarfile
from pathlib import Path
from collections import OrderedDict
import requests
from requests.adapters import HTTPAdapter, Retry

TCGA_API_BASEURL = 'https://api.gdc.cancer.gov'
TCGA_CASE_ENDPOINT = '/'.join([TCGA_API_BASEURL, 'cases'])
TCGA_PROJECT_ENDPOINT = '/'.join([TCGA_API_BASEURL, 'projects'])
TCGA_DATA_ENDPOINT = '/'.join([TCGA_API_BASEURL, 'data'])
TCGA_FILE_ENDPOINT = '/'.join([TCGA_API_BASEURL, 'files'])

# Metadata parts
def _get_metadata_from_tcga_api(endpoint, **kwargs):
    '''
    Get the metadata that return from the TCGA API.

    :param endpoint: Specify the endpoint.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    params = kwargs

    for param in params:
        if type(params[param]) == list:
            params[param] = ','.join(params[param])

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.get(endpoint, params=params)
    response.raise_for_status()

    return response.json()

def get_metadata_from_project(project_id, **kwargs):
    '''
    Get the response that return from the specific project_id using arguments.

    :param project_id: Specify the project that you want.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    tcga_project_id_endpoint = '/'.join([TCGA_PROJECT_ENDPOINT, project_id])

    return _get_metadata_from_tcga_api(endpoint=tcga_project_id_endpoint, **kwargs)['data']

def get_metadata_from_case(case_id, **kwargs):
    '''
    Get the response that return from the specific case_id using arguments.

    :param case_id: Specify the case that you want.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    tcga_case_id_endpoint = '/'.join([TCGA_CASE_ENDPOINT, case_id])

    return _get_metadata_from_tcga_api(endpoint=tcga_case_id_endpoint, **kwargs)['data']

def get_metadata_from_file(file_id, **kwargs):
    '''
    Get the response that return from the specific file_id using arguments.

    :param file_id: Specify the file that you want.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    tcga_file_id_endpoint = '/'.join([TCGA_FILE_ENDPOINT, file_id])

    return _get_metadata_from_tcga_api(endpoint=tcga_file_id_endpoint, **kwargs)['data']

# Filter parts 
def _build_query_pairs(query_dict):
    '''
    Build the field and value for each operation.

    :param query_dict: Query pairs
    '''
    temp_dict = OrderedDict()

    if len(query_dict.items()) > 1:
        raise ValueError(f'One operation has only one field, get {len(query_dict.itemrs())} fields.')

    field, value = list(query_dict.items())[0]
    
    temp_dict['field'] = field
    temp_dict['value'] = value

    return temp_dict

def _build_filters(filters):
    '''
    Build the filters for each request using simple dictionary pairs.

    :param filters: The filters you want to use. Please see the GDC API documentations.
    '''
    encoded_filters = OrderedDict()

    valid_operators = ['=', '!=', '<', '<=', '>', '>=', 'is', 'not', 'in', 'exclude', 'and', 'or']
    for operator in filters:
        if operator not in valid_operators:
            raise KeyError(f'Invalid operator {operator}.')

        encoded_filters['op'] = operator
        if operator not in ['and', 'or']:
            encoded_filters['content'] = _build_query_pairs(filters[operator])
        else:
            contents = []
            for sub_filter in filters[operator]:
                content = json.loads(_build_filters(sub_filter))
                contents.append(content)

            encoded_filters['content'] = contents

    return json.dumps(encoded_filters)

def _get_filters_result_from_tcga_api(endpoint, filters=None, **kwargs):
    '''
    Get the result with filters that return from the TCGA API.

    :param endpoint: Specify the endpoint.
    :param filters: The filters that you want to search. Please see the GDC API documentations.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    params = kwargs

    if filters is not None:
        encoded_filters = _build_filters(filters)
        params['filters'] = encoded_filters

    for param in params:
        if type(params[param]) == list:
            params[param] = ','.join(params[param])

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.get(endpoint, params=params)
    response.raise_for_status()

    return response.json()

def get_filters_result_from_project(**kwargs):
    '''
    Get the result with filters that return from the project endpoint.

    :param filters: The filters that you want to search. Please see the GDC API documentations.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    return _get_filters_result_from_tcga_api(endpoint=TCGA_PROJECT_ENDPOINT, **kwargs)['data']['hits']

def get_filters_result_from_case(**kwargs):
    '''
    Get the result with filters that return from the cases endpoint.

    :param filters: The filters that you want to search. Please see the GDC API documentations.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    return _get_filters_result_from_tcga_api(endpoint=TCGA_CASE_ENDPOINT, **kwargs)['data']['hits']

def get_filters_result_from_file(**kwargs):
    '''
    Get the result with filters that return from the file endpoint.

    :param filters: The filters that you want to search. Please see the GDC API documentations.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    return _get_filters_result_from_tcga_api(endpoint=TCGA_FILE_ENDPOINT, **kwargs)['data']['hits']

# Download parts
# Add controlled access token
# Think how to handle different methods
def _get_download_from_tcga_api(file_ids, **kwargs):
    '''
    Download the files using GET method with file_ids. Recommended for small number of files to download.

    :param file_ids: A list of file_ids to download. Please see the GDC API documentations.
    :param **kwargs: The parameters that you want to add in the download. Please see the GDC API documentations for the valid parameters.
    '''
    tcga_file_id_endpoint = '/'.join([TCGA_DATA_ENDPOINT, ','.join(file_ids)])

    headers = {
        'Content-Type': 'application/json'
    }

    params = kwargs

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.get(
        tcga_file_id_endpoint,
        headers=headers,
        params=params
    )
    response.raise_for_status()

    response_head_cd = response.headers['Content-Disposition']

    file_name = re.findall("filename=(.+)", response_head_cd)[0]

    return response.content, file_name

def _post_download_from_tcga_api(file_ids, **kwargs):
    '''
    Download the files using POST method with file_ids. Recommended for large number of files to download.

    :param file_ids: A list of file_ids to download. Please see the GDC API documentations.
    :param **kwargs: The parameters that you want to add in the download. Please see the GDC API documentations for the valid parameters.
    '''
    data = {'ids': file_ids}

    headers = {
        'Content-Type': 'application/json'
    }

    params = kwargs

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[500, 502, 504], allowed_methods=frozenset(['GET', 'POST']))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.post(
        TCGA_DATA_ENDPOINT,
        data=json.dumps(data),
        headers=headers,
        params=params
    )
    response.raise_for_status()

    response_head_cd = response.headers['Content-Disposition']

    file_name = re.findall("filename=(.+)", response_head_cd)[0]

    return response.content, file_name

#TODO
# Fix the single file with related files
# Add the progress bar
def download_file(file_id, extract_directory, method='GET', **kwargs):
    '''
    Download a single file using a file_id.

    :param file_id: A file_id to download. Please see the GDC API documentations.
    :param extract_directory: The directory that you wanted to store the files.
    :param method: The method you wanted request to download the file.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    if method == 'GET':
        download_from_tcga_api = _get_download_from_tcga_api
    elif method == 'POST':
        download_from_tcga_api = _post_download_from_tcga_api
    else:
        raise KeyError('Wrong methods. Only accept GET and POST')

    content, file_name = download_from_tcga_api([file_id], **kwargs)

    with open('/'.join([extract_directory, file_name]), 'wb') as output_file:
        output_file.write(content)

#TODO
# Add the progress bar
def download_files(file_ids, extract_directory, method='POST', **kwargs):
    '''
    Download multiple file using a list of file_ids.
    Get the result with filters that return from the cases endpoint

    :param file_ids: A list of file_ids to download. Please see the GDC API documentations.
    :param extract_directory: The directory that you wanted to store the files.
    :param method: The method you wanted request to download the files.
    :param **kwargs: The parameters that you want to add in the request. Please see the GDC API documentations for the valid parameters.
    '''
    if method == 'GET':
        download_from_tcga_api = _get_download_from_tcga_api
    elif method == 'POST':
        download_from_tcga_api = _post_download_from_tcga_api
    else:
        raise KeyError('Wrong methods. Only accept GET and POST')

    content, _ = download_from_tcga_api(file_ids, **kwargs)

    file_like_object = io.BytesIO(content)
    
    with tarfile.open(fileobj=file_like_object, mode='r:gz') as tar:

        tar.extractall(path=extract_directory)
        file_names = [Path('/'.join([extract_directory, file_name])) for file_name in tar.getnames() if file_name not in ['MANIFEST.txt']]

    for file_name in file_names:
        file_name.rename('/'.join([extract_directory, file_name.name]))
        file_name.parent.rmdir()

    os.remove('/'.join([extract_directory, 'MANIFEST.txt']))
