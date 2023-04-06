from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
#Setup of flask
app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#Routing the htmlwebpage using route function
@app.route('/',methods=['GET','POST'])
def index():
    path="mushrooms.csv"
    my_data = pd.read_csv("mushrooms.csv", delimiter=",")
    X = my_data[['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-spacing','gill-size','gill-color']].values
    
    le_shape = preprocessing.LabelEncoder()
    le_shape.fit(['b','c','x','f','k','s'])
    X[:,0] = le_shape.transform(X[:,0]) 


    le_surface = preprocessing.LabelEncoder()
    le_surface.fit([ 'f', 'g', 'y','s'])
    X[:,1] = le_surface.transform(X[:,1])


    le_color = preprocessing.LabelEncoder()
    le_color.fit([ 'n','b','c','g','r','p','u','e','w','y'])
    X[:,2] = le_color.transform(X[:,2])

    le_odor = preprocessing.LabelEncoder()
    le_odor.fit([ 'a', 'l', 'c','y','f','m','n','p','s'])
    X[:,3] = le_odor.transform(X[:,3])

    le_gspace = preprocessing.LabelEncoder()
    le_gspace.fit([ 'c', 'w', 'd'])
    X[:,4] = le_gspace.transform(X[:,4])

    le_gsize = preprocessing.LabelEncoder()
    le_gsize.fit([ 'b', 'n'])
    X[:,5] = le_gsize.transform(X[:,5])

    le_gcolor = preprocessing.LabelEncoder()
    le_gcolor.fit([ 'k', 'n', 'b','h','g','r','o','p','u','e','w','y'])
    X[:,6] = le_gcolor.transform(X[:,6])

    y = my_data["class"]
    from sklearn.model_selection import train_test_split
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
    drugTree.fit(X_trainset,y_trainset)
    predTree = drugTree.predict(X_testset)
    print(drugTree.predict([['2','3' ,'0' ,'7','0','1','0' ]]))
    cap={
        "bell-b":'0',
        "conical-c":'1',
        "convex-x":'2',
        "flat-f":'3',
        "knobbed-k":'4',
        "sunken-s":'5',
        "fibrous-f":'0',
        "grooves-g":'1',
        "scaly-y":'2',
        "smooth-s":'3',
        "brown-n":'0',
        "buff-b":'1',
        "cinnamon-c":'2',
        "gray-g":'3',
        "green-r":'4',
        "pink-p":'5',
        "purple-u":'6',
        "red-e":'7',
        "white-w":'8',
        "yellow-y":'9',
        "almond-a":'0',
        "anise-l":'1',
        "creasote-c":'2',
        "fishy-y":'3',
        "foul-f":'4',
        "mustly-m":'5',
        "none-n":'6',
        "pungent-p":'7',
        "spicy-s":'8'
    }
    gill={
        "close-c":'0',
        "crowded-w":'1',
        "distant-d":'2',
        "broad-b":'0',
        "narrow-n":'1',
        "black-k":'0',
        "brown-n":'1',
        "buff-b":'2',
        "chocolate-h":'3',
        "gray-g":'4',
        "green-r":'5',
        "orange-o":'6',
        "pink-p":'7',
        "purple-u":'8',
        "red-e":'9',
        "white-w":'10',
        "yellow-y":'11'
    }
    db.drop_all()
    db.create_all()
    if request.method=='POST':
        capshape=request.form['capshape']
        capsurface=request.form['capsurface']
        capcolor=request.form['capcolor']
        odor=request.form['odor']
        gillspacing=request.form['gillspacing']
        gillsize=request.form['gillsize']
        gillcolor=request.form['gillcolor']
        a=drugTree.predict([[cap[capshape],cap[capsurface] ,cap[capcolor] ,cap[odor],gill[gillspacing],gill[gillsize],gill[gillcolor]]]) 
        return render_template('mushroom.html',result=a[0])
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True,port=8000)