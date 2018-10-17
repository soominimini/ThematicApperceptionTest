
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from konlpy.tag import Twitter
from konlpy.tag import Kkma
import  nltk
import app

result = Flask(__name__)
result.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(result)



if __name__ == '__main__':
    result.run()
