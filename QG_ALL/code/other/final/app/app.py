import json
from flask import Flask, jsonify, request
import sys
import numpy as np
sys.path.append("..")
from Model.check import CheckData
from Model.HSB_Algorithm import HSB
from Model.MWMS_J_Algorithm import MWMS_J
from Model.MWMS_S_Algorithm import MWMS_S
from Model.DSG_Algorithm import DSG
from Model.RSRSP_Algorithm import RSRSP

app = Flask(__name__)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=32617)
    global model, x, file, algorithm, DP, rc


@app.route('/pushfile', methods=['GET', 'POST'])
def push_file():
    global model, x, file, algorithm, DP, rc
    _dict = json.loads(request.get_data(as_text=True))
    file = _dict["file"]
    algorithm = _dict["algorithm"]
    DP = True if _dict["DP"] == "True" else False
    rc = _dict["rc"]
    checkData = CheckData(file, rc)
    x = checkData.main()
    status_dict = {
        1: "The format of the file you give me must be csv, xls, xlsx, npy, mat or txt!",
        2: "The content of the file must be a number matrix. Please check it again!",
        3: "The parameter you give me must be a positive number! Please check it again!"
                   }
    success_str = "The file you give me is a %d X %d matrix. Are you sure use it to train?"
    if type(x) == int:
        return json.dumps({'msg': status_dict[x], 'code': x+410})
    else:
        return json.dumps({'msg': success_str % (np.shape(x)), 'code': x+600})

@app.route('/run')
def train():
    global model, x, file, algorithm, DP, rc
    if algorithm == "HAB":
        model = HSB(DP_flag=DP, rc=rc)
    elif algorithm == "DSG":
        model = DSG(DP_flag=DP, rc=rc)
    elif algorithm == "MWMS-J":
        model = MWMS_J(DP_flag=DP, rc=rc)
    elif algorithm == "MWMS-S":
        model = MWMS_S(DP_flag=DP, rc=rc)
    elif algorithm == "RSRSP":
        model = RSRSP(DP_flag=DP, rc=rc)
    # model.main(x)
    print("I am running!")
