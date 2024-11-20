import random
import time
def current_wait_time(hospital: str)->int |str:
    "function to create fakw wait times"
    if hospital not in ["A", "B", "C", "D"]:
        return f"Hospital {hospital} does not exist"
    time.sleep(1)
    return random.randint(0,1000)
