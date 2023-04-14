import joblib
import numpy as np
import pandas as pd
import torch
from car_predict_model import ANNPredictionModel, XGBoost, DTR, RFR


def print_options(options):
    for i, option in enumerate(options):
        print(f"{i + 1}. {option}")


def main():
    manufacturers = "nissan,mercedes-benz,mazda,volvo,ford,infiniti,audi,alfa-romeo,gmc,hyundai,dodge,acura,chevrolet,jeep,bmw,toyota,cadillac,lexus,ram,volkswagen,jaguar,honda,kia,subaru,mitsubishi,chrysler,mini,lincoln,pontiac,buick,porsche,rover,fiat,mercury,saturn,tesla,harley-davidson,land rover,ferrari,aston-marti".split(
        ',')
    fuels = "gas,hybrid,diesel,electric,other".split(',')
    title_statuses = "clean,rebuilt,lien,missing,parts only,salvage".split(',')
    transmissions = "automatic,manual,other".split(',')
    types = "sedan,SUV,wagon,truck,hatchback,pickup,coupe,mini-van,convertible,van,offroad,bus,other".split(',')
    states = "wa,nj,wi,co,ca,mi,az,ky,vt,oh,al,or,dc,tx,fl,pa,tn,ga,va,de,ia,nc,ny,il,ut,ok,mo,ks,nv,hi,id,ct,ak,mn,ne,ma,wv,ri,in,sd,la,md,sc,me,ar,mt,nm,nh,ms,wy,nd".split(
        ',')
    year_default = 2012
    odometer_default = 95854.51697463683

    print_options(manufacturers)
    manufacturer = int(input("Please select your manufacturer from the list: "))
    manufacturer = manufacturers[manufacturer - 1]

    year = input("Please enter the year of your car (default 2012): ")
    if year.strip() == "":
        year = year_default
    year = float(year)

    odometer = input("Please enter the odometer of your car (default 95854.51697463683): ")
    if odometer.strip() == "":
        odometer = odometer_default
    odometer = float(odometer)

    print_options(fuels)
    fuel = int(input("Please select your fuel from the list: "))
    fuel = fuels[fuel - 1]

    print_options(title_statuses)
    title_status = int(input("Please select your title status from the list: "))
    title_status = title_statuses[title_status - 1]

    print_options(transmissions)
    transmission = int(input("Please select your transmission from the list: "))
    transmission = transmissions[transmission - 1]

    print_options(types)
    type = int(input("Please select your type from the list: "))
    type = types[type - 1]

    print_options(states)
    state = int(input("Please select your state from the list: "))
    state = states[state - 1]

    columns = ['year', 'manufacturer', 'fuel', 'odometer', 'title_status', 'transmission', 'type', 'state']
    X = pd.DataFrame([[year, manufacturer, fuel, odometer, title_status, transmission, type, state]], columns=columns)
    print(X.shape)
    models = ['ANN', 'XGBoost', 'decision_tree', 'random_forest']
    print_options(models)
    model_name = int(input("Please select your model from the list: "))
    model_name = models[model_name - 1]

    model = None
    if model_name == 'ANN':
        model = ANNPredictionModel('ann/model/ann_best.ckpt')
    elif model_name == 'XGBoost':
        model = XGBoost('xgboost/xgb_model.joblib')
    elif model_name == 'decision_tree':
        model = DTR('decision_tree_regressor/decision_tree_regressor.joblib')
    elif model_name == 'random_forest':
        model = RFR('random_forest_regressor/random_forest_regressor.joblib')
    y = model.predict(X)
    print("The predicted price is: ", y)


if __name__ == '__main__':
    main()
