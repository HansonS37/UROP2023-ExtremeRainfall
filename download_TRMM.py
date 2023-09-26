from datetime import date, timedelta
from pydap.client import open_url
from pydap.cas.urs import setup_session
import requests

# read from the txt file
urls = open('subset_TRMM_3B42_Daily_7_20230703_152008_.txt','r')
url_con = urls.read()
urls.close()
lines = url_con.splitlines()
lines = lines[1:]
#time = date(1998,1,1)
#index = 0
time = date(1998,1,1)
for l in lines:
    #print(index)
    URL = l
# Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name. 
    print(time.strftime("%Y%m%d"))
    FILENAME = time.strftime("%Y%m%d") + '.nc4'
    result = requests.get(URL)
    try:
        result.raise_for_status()
        f = open(FILENAME, 'ab')
        f.write(result.content)
        f.close()
        print('contents of URL written to '+FILENAME)
    except:
        print('requests.get() returned an error code '+str(result.status_code))
    #index += 1
    time = time + timedelta(days=1)


"""print(lines[1])
dataset_url = lines[1]

username = 'eliashanbiip'
password = 'i4FD@#FLj*93i5m'

try:
    session = setup_session(username, password, check_url=dataset_url)
    dataset = open_url(dataset_url, session=session)
except AttributeError as e:
    print('Error:', e)
    print('Please verify that the dataset URL point')"""