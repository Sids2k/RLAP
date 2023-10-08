import pickle
import sklearn.ensemble as se
from sklearn.metrics import classification_report
import xgboost as xgb
import numpy as np
import joblib
version = 3

XColl,yColl,XColltest,yColltest = pickle.load(open(f"demo_data/XYXtYtCollv{version}.pkl","rb"))

# print("XColl", XColl[0])
def newFeatures(x):
    newX = x[:4]
    for a in x[:4]:
        for b in x[:4]:
            if a < b:
                newX.append(a-b)
                # newX.append(b-a) (doesn't change the result)
                # newX.append(a+b)
            # else:
                # print("NOOO", a, b)
    newX.extend(x[4:])
    return newX
# print("XColl", XColl[0])
# print("NewXColl", newFeatures(XColl[0]))


XColl = np.array([newFeatures(x) for x in XColl])
XColltest = np.array([newFeatures(x) for x in XColltest])
# pmodel = se.RandomForestClassifier(n_estimators=1000, max_depth=500,random_state=0, verbose=1, n_jobs=-1)
pmodel = xgb.XGBClassifier(n_estimators=3000, max_depth=50,random_state=2, verbose=1, n_jobs=-1)
# pmodel = se.RandomForestClassifier(n_estimators=100, max_depth=50,random_state=0, verbose=0, n_jobs=-1)
pmodel.fit(XColl,yColl)



print("pmodel")
print(pmodel.score(XColl,yColl),pmodel.score(XColltest,yColltest))
print(sum(pmodel.predict(XColltest)), sum(yColltest))
# find f1 score as pmodel on xcoll and xcolltest
target_names = ['feasible', 'not feasible']
print(classification_report(yColl, pmodel.predict(XColl),target_names = target_names))
print(classification_report(yColltest, pmodel.predict(XColltest),target_names = target_names))

joblib.dump(pmodel, f"models/pmodelv{version}.joblib")


exit()