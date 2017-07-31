import requests
import random
from build_model import predict
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
company_name = 'framgia'

for i in range(100):
    s = requests.Session()

    s.get('https://chuyencuadev.com/', verify=False)
    r = s.get(f'https://chuyencuadev.com/{company_name}/reviews', verify=False)

    # print(r.text)


    import time

    current_milli_time = lambda: int(round(time.time() * 1000))
    r = s.get("https://chuyencuadev.com/captcha?" + str(current_milli_time()))

    with open('captcha.jpg', 'wb') as f:
        f.write(r.content)

    headers = {
        "accept":"*/*",
        "accept-language":"en-US,en;q=0.8,ja;q=0.6,vi;q=0.4",
        "content-type":"application/x-www-form-urlencoded; charset=UTF-8",
        "dnt":"1",
        "origin":"https://chuyencuadev.com",
        "user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.78 Safari/537.36",
        "x-requested-with":"XMLHttpRequest",
    }
    salary_benefit_rate = random.randrange(3,6,1)
    training_learning_rate = random.randrange(3,6,1)
    management_care_rate = random.randrange(3,6,1)
    culture_fun_rate = random.randrange(3,6,1)
    office_workspace_rate = random.randrange(3,6,1)
    payload = {
        "salary_benefit_rate":f"{salary_benefit_rate}",
        "training_learning_rate":f"{training_learning_rate}",
        "management_care_rate":f"{management_care_rate}",
        "culture_fun_rate":f"{culture_fun_rate}",
        "office_workspace_rate":f"{office_workspace_rate}",
    }
    payload["captcha"] = predict('captcha.jpg')
    # raw_input().strip()
    # print(payload)

    r = s.post(f"https://chuyencuadev.com/{company_name}/review", headers=headers, data=payload)