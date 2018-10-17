import os
import random

from Cython.Tempita._tempita import url
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
#from pandas.tests.test_base import T
from scipy.io.matlab.mio5_params import mat_struct
from werkzeug.utils import secure_filename

#한글 형태소 분석을 위한 패키지 import
from konlpy.tag import Twitter
from konlpy.tag import Kkma
from gensim.models import Word2Vec
import re
import pandas as pd
import numpy as np


#결과를 워드클라우드 형태로 보여주기 위한 패키지 import
from collections import Counter
import pytagcloud

import time

#sys는 변수 동적 생성을 위해 임포트
import sys
mod: object = sys.modules[__name__]

#Analyzes text using the Google Cloud Natural Language API

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)


#DB 구조 만듦
#각각의 클래스로 따로 만들어도 되지만 하나의 Picture클래스만 만들어서 그아래에서 tablename을 각각으로 둬서 만들어도 됨
class Male(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20))

    def __init__(self,id = None, name=None):
        self.id= id
        self.name = name

class Female(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20))

    def __init__(self,id = None,  name=None):
        self.id = id
        self.name = name

class Child(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20))

    def __init__(self,id = None,  name=None):
        self.id = id
        self.name = name

    #문자열 DB 남자문자열, 여자문자열, 아동문자열
class MaleString(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    descript = db.Column(db.VARCHAR(150))

    def __init__(self, id = None, descript=None):
        self.id = id
        self.descript = descript

class FemaleString(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    descript = db.Column(db.VARCHAR(150))

    def __init__(self,id = None,  descript=None):
        self.id = id
        self.descript = descript

class ChildString(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    descript = db.Column(db.VARCHAR(150))

    def __init__(self, id = None, descript=None):
        self.id = id
        self.descript = descript

class TimeDB(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timeCount = db.Column(db.Integer)

    def __init__(self, id = None, timeCount=None):
        self.id = id
        self.timeCount = timeCount





#API 사용 함수

def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))
        setattr(mod, 'var_{}'.format(index, sentence_sentiment), index) #동적 변수 생성
        render_template('google_result.html', GCP='var_{}')


    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))


def analyze(i):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    male_descript = MaleString.query.get(i).descript
    content = male_descript
    #with open("GCP.txt", 'r') as fileForGCP:
        # Instantiates a plain text document.
       # content = fileForGCP.read()



    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)

    # Print the results
    print_result(annotations)
    return render_template('google_result.html', GCP=annotations)
#annotation, 위의 함수에서는 sentence.sentiment.score을 인자로 넣었는데 둘 다 int형이어서 콜을 할 수가 없음
#int 형은 callable 하지 않다고 에러가 뜸
#The return type must be a string, tuple, Response instance, or WSGI callable



@app.route('/')
def index():
    return render_template('index.html')

#각각 해당하는 이미지 파일들을 보여주는 목록 페이지
@app.route('/male/')
def male_list():
    male = Male.query.all()
    return render_template('male_list.html', male = male)

@app.route('/female/')
def female_list():
    female = Female.query.all()
    return render_template('female_list.html', female = female)

@app.route('/child/')
def child_list():
    child = Child.query.all()
    return render_template('child_list.html', child = child)


# 실제 이미지 파일을 보여주는 페이지
@app.route('/maleTest/<id>', methods=['GET', 'POST'])
def male_detail(id):
    if request.method=="GET":
        db.session.commit()
        global start
        start =time.time()
        print(start)
        homme = Male.query.get(id) # Picture.query.all(id = 1).first() 같은 방식으로 가져올 수도 있음
        #여기서 넘어가는 거는 id가 넘어가는게 아니라 Male 객체 하나가 그대로 넘어가는 것임
        return render_template('maleIMG.html', male = homme)

    elif request.method=="POST":
        e = int(time.time() - start)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

     #  time_mesure(e)
        timedb = TimeDB()
        time_id = Male.query.get(id).id
        timedb.id = time_id
        timedb.timeCount =e
        db.session.add(timedb)
        db.session.commit()
        #시간 DB 객체 생성하고 데이터 저장

        #form 태그에서 url_for로 정해주지 않아도 해당 페이지 내에서는 POST 방식으로 들어왔을 경우에 실행하는 코드르 짜 놨으니 action = "url_for()"가 없어도 동작함
        # 페이지 하나씩 차례로 넘기는 코드
        strD = MaleString()
        strD.id = id
        strD.descript = request.form['txt']
        db.session.add(strD)
        db.session.commit()
        # 문자열 DB에 저장하고 페이지 넘기기
        id = str(int(id) + 1)
        homme = Male.query.get(id)
        return render_template('maleIMG.html', male=homme)


# 실제 이미지 파일을 보여주는 페이지
@app.route('/femaleTest/<id>', methods=['GET', 'POST'])
def female_detail(id):
    if request.method=="GET":
        female = Female.query.get(id) # Picture.query.all(id = 1).first() 같은 방식으로 가져올 수도 있음
        return render_template('femaleIMG.html', female = female)
    elif request.method=="POST":
        # 페이지 하나씩 차례로 넘기는 코드
        strF = FemaleString()
        strF.id = id

        strF.descript = request.form['txt']
        db.session.add(strF)
        db.session.commit()
        # 문자열 DB에 저장하고 페이지 넘기기
        id = str(int(id) + 1)
        femme = Male.query.get(id)
        return render_template('femaleIMG.html', female=femme)

# 실제 이미지 파일을 보여주는 페이지
@app.route('/childTest/<id>', methods=['GET'])
def child_detail(id):
    child = Child.query.get(id) # Picture.query.all(id = 1).first() 같은 방식으로 가져올 수도 있음
    return render_template('childIMG.html', child=child)


#새로운 글 쓰는 페이지
@app.route('/numbers/maleNew', methods=['GET', 'POST'])
def writeMale():
    if request.method == "GET":
        return render_template("create.html")

    elif request.method == "POST":
        male = Male()
        male.id  = request.form['title']
        image = request.files['image']
        image_name = secure_filename(image.filename)  #use this function to secure a filename before storing it directly on the filesystem.

        image.save('male/' + image_name)  # image saving( filepath , image name)
        #use the save() method of the file to save the file permanently somewhere on the filesystem

        male.name = image_name # database insert
        db.session.add(male)
        db.session.commit()

        return redirect(url_for('male_list'))  # url_for 뒤에는 html파일 이름이 오는 게 아니라 함수이름이 온다


@app.route('/numbers/femaleNew', methods=['GET', 'POST'])
def writeFemale():
        if request.method == "GET":
            return render_template("create.html")

        elif request.method == "POST":
            female = Female()
            female.id = request.form['title']
            image = request.files['image']
            image_name = secure_filename(image.filename)  # use this function to secure a filename before storing it directly on the filesystem.

            image.save('female/' + image_name)  # image saving( filepath , image name)
            # use the save() method of the file to save the file permanently somewhere on the filesystem

            female.name = image_name  # database insert
            db.session.add(female)
            db.session.commit()

            return redirect(url_for('female_list'))  # url_for 뒤에는 html파일 이름이 오는 게 아니라 함수이름이 온다

@app.route('/numbers/childNew', methods=['GET', 'POST'])
def writeChild():
            if request.method == "GET":
                return render_template("create.html")

            elif request.method == "POST":
                child = Child()
                child.id = request.form['title']
                image = request.files['image']
                image_name = secure_filename(
                    image.filename)  # use this function to secure a filename before storing it directly on the filesystem.

                image.save('child/' + image_name)  # image saving( filepath , image name)
                # use the save() method of the file to save the file permanently somewhere on the filesystem

                child.name = image_name  # database insert
                db.session.add(child)
                db.session.commit()

                return redirect(url_for('child_list'))  # url_for 뒤에는 html파일 이름이 오는 게 아니라 함수이름이 온다



@app.route('/time')
def time_mesure():
    len = TimeDB.query.count()
    timeClass = TimeDB.query.all()
    for timeData in range(len):
        TimeID = TimeDB.query.get(timeData+1).id
        TimeCount = TimeDB.query.get(timeData+1).timeCount
        print(TimeID)
        print(TimeCount)
        print('{:02d}:{:02d}:{:02d}'.format(int(TimeCount) // 3600, (int(TimeCount) % 3600 // 60), int(TimeCount) % 60))

    return render_template('timeTest.html', timeClass= timeClass)


@app.route('/male/<path:name>')
def download_fileMale(name):
    return send_from_directory('male', name)  # send_from_directory(app.static_folder, filename)

@app.route('/female/<path:name>')
def download_fileFemale(name):
    return send_from_directory('female', name)  # send_from_directory(app.static_folder, filename)

@app.route('/child/<path:name>')
def download_fileChild(name):
    return send_from_directory('child', name)  # send_from_directory(app.static_folder, filename)


#영어 문자로 넣은 것은 결과로 못 돌아감
#UserWarning: C extension not loaded, training will be slow. Install a C compiler and reinstall gensim for fast training 이런 에러 뜸
@app.route('/result')
def final():
    t = Twitter()
    k = Kkma()
    mStr = MaleString.query.count()      #mStr에 int형으로 개수가 들어감
    mStrDesc = MaleString.query.get(1).descript
    # 이렇게 Twitter(), Kkma() 는 함수 안에서 생성해 주어야 함 외부에서 전역 변수로 생성해 주면 exit-code-0xc0000005 에러 남
    # access violation 에러
    
    total = k.nouns(MaleString.query.get(1).descript)
    for i in range(1,mStr):
        i += 1
        strTest = MaleString.query.get(i).descript
        countTest = k.nouns(strTest)
        total.extend(countTest)

    print(total)
    count = Counter(total)
    print(count)
    tags2 = count.most_common(15)
    taglist = pytagcloud.make_tags(tags2, maxsize=50)

    fileStr = MaleString.query.get(1).descript
    for i in range(1,mStr):
        i += 1
        strTest = MaleString.query.get(i).descript
        fileStr+=strTest

    #word2vec을 사용하기 위한 파일 : TAT_stringTest.txt
    with open("TAT_stringTest.txt",'w')as f:
        "".join(fileStr) #리스트에서 문자열으로
        f.write(fileStr)


    file = open("TAT_stringTest.txt", 'r')
    readFile = file.read()
    readFile = re.split("[\n\.?]", fileStr) #여기서 리스트 형태로 변환됨 comma를 기준으로  나눔
    while '' in readFile:
        readFile.remove('')

    entireString = pd.DataFrame()
    entireString['sentence'] = np.asarray(readFile)
    entireString['sentence_seperated'] = entireString['sentence'].apply(lambda x: x.replace(",", ""))
    entireString['sentence_seperated'] = entireString['sentence'].apply(lambda x: x.replace(";", ""))
    entireString['sentence_seperated'] = entireString['sentence'].apply(lambda x: x.replace("\n", ""))
    entireString['sentence_seperated'] = entireString['sentence'].apply(lambda x: x.split())

    model = Word2Vec(entireString['sentence_seperated'], hs=1, window=2, size=300, min_count=1)
    # hs = 1 hs는 int {1,0}  만약에 1이 오면, hierarchical softmax will be used for model training.
    # hs = 1 hs는 int {1,0}  만약에 1이 오면, hierarchical softmax will be used for model training.
    # I만약에 0으로 설정되고 navigate가 non-zero이면 negative sampling will be used.
    # min_count =5등장 횟수가 5 이하인 단어는 무시
    # sizs = 300 300차원짜리 벡터스페이스에 embedding
    print(model)
    #you must first build vocabulary before training the model의 의미는 min_count로 설정되어 있는 것 만큼 많이 나타난 단어가 없다는 뜻


    for word, score in model.most_similar("아이는"):
        print(word)
   # print(model.most_similar('남자는', top=5))

    #"word '여자' not in vocabulary"  keyerr가 계속 나는데 이유가 most_similar로 나올만한 단어가 없는 듯함
    # 그런데 '여자는, 여성이, 아내가, 아내는'이라는 식으로 조사와 함께 붙여서 검색해보면 결과가 나오기는 함
    #아무래도 영어와 달리 한국어는 조사가 명사에 붙기 때문에 영어에서는 'whale'이라고만 검색을 하면 결과가 나오는 것과 달리
    # 한국어 검색에서는 조사를 붙여서 검색을 해야함, 그런데 조사가 달라지면 결과도 달라지는 문제가 있음



    pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', size=(900, 600), fontname='Korean', rectangular=False)

    return render_template('result.html', mStr = total)

    if len(total) >1:
        del total[:]


@app.route('/google_result')
def gcp():
    fileStr = MaleString.query.get(1).descript
    #GCP를 사용하기 위한 파일(쓰기) : GCP.txt
    with open("GCP.txt",'w')as fileForGCP:
        "".join(fileStr) #리스트에서 문자열으로
        fileForGCP.write(fileStr)


    maleString_total_count = MaleString.query.count()


    for i in range(1,maleString_total_count+1 ):
        analyze(i)
        i+=1
    # entities sentiment analysis는 한국어 지원이 안돼서 기능 안넣었음
    return render_template('google_result.html')



@app.route('/show')
def show_all():
#    mStr = MaleString.query.count()
#    all_descript = ""
#    for i in range(mStr):
#        i += 1
#        strTest = MaleString.query.get(i).descript +'\n'
#        all_descript +=strTest
#    print(all_descript)
    all_descript = MaleString.query.all()
    return render_template('show.html', descript=all_descript)



if __name__ == '__main__':
    app.run(debug=True)

