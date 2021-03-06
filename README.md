# ThematicApperceptionTest

>> ***해당 페이지는 2018년 11월 학술대회에서 발표한 논문의 내용을 축약해서 작성한 것입니다.***

## 요약

현재 국내에 존재하는 대부분의 심리 검사 어플리케이션은 간단한 다지선다 응답 형 심리테스트이거나 성인들을 위한 진로 및 적성 검사와 성격 유형 검사에 그친다. 본 연구에서는 사용자로부터 서술형 문장을 입력 받은 후 자연어 처리를 통해 객관적으로 결과를 분석해 주는 투사적 심리 검사 시스템을 개발하였다. 
본 연구에서 개발된 프로그램은 사용자가 웹에서 검사를 시행한 후 피검자의 검사 내용을 데이터베이스에 저장 및 관리하며 구글 클라우드 플랫폼의 자연어 처리 기술을 사용하여 검사 내용을 분석하고 검사자에게 결과를 보여주는 시스템이다. 이를 통해 검사자는 검사 결과 해석 시에 자연어 처리 과정을 통해 분석한 결과를 참고함으로써 자신의 결과 해석에 객관성을 부여할 수 있으며 검사자가 직접 해석을 하기 전에 컴퓨터의 해석 결과를 확인함으로써 전체적인 검사 결과 해석 시간을 단축 시키는 효율적인 측면이 있다.      




### 심리 검사 시스템 설계
 시스템 전체 구성은 그림 1과 같다. 여기서 전처리 과정은 피검자가 웹 페이지에서 검사를 한 후 결과를 DB에 저장하는 단계까지이고, 후처리 과정은 DB에 입력된 문자 데이터를 구글 NLP API를 통해 분석하는 단계이다.



그림 1) 시스템 흐름도
3.1 전처리 과정

<div>
 
<img width = "500" src ="https://user-images.githubusercontent.com/28712478/52125197-0c3c1400-266f-11e9-951c-293d20b0245e.png">

</div>


<div>
<img width="500" src = "https://user-images.githubusercontent.com/28712478/52125654-7d2ffb80-2670-11e9-9a56-ff419f30daa6.png">
 
 
 (그림 2) 검사 페이지   
</div>
   
그림 1에서 보이는 바와 같이 피검자는 처음 웹 페이지에 들어갔을 때 자신의 성별을 선택한 후 검사를 진행한다. 피검자는 각 페이지마다 그림 2와 같은 검사 페이지를 보며 자신만의 이야기를 만든 후 그림 아래에 있는 textarea에 작성한다. 이후 다음 페이지로 이동하기 전에 시스템은 피검자로부터 텍스트가 들어왔는지 확인한 후 만일 들어오지 않았을 경우에는 문자 입력을 요하는 알림을 띄우고 페이지 이동을 막는다. 그러나 피검자에게서 텍스트가 들어왔을 경우에는 피검자의 텍스트를 stringDB에 각 문항 별로 저장한다. 또 피검자가 1번 문항에 들어온 시간부터 2번 문항으로 이동하기 전까지 걸린 시간을 Time package를이용하여 시간을 재고 이를 time DB에 저장한다. 시간 확인 페이지를 통해 검사자는 피검자가 검사에 응하는 시간이 전반적으로 얼마나 걸렸는지 확인할 수 있다. 뿐만 아니라 구체적으로 피검자가 어떤 문항에서 응답에 오랜 시간이 걸렸는지 또한 확인이 가능하다.






### 3.2 후처리 과정
 DB에 있는 문장 분석을 위해서 word2vec함수와 AI API를 사용하였다. 우선 Word2vec 함수는 단어의 의미를 벡터화 한 것으로 각 단어들 사이의 유사도를 측정한다. 즉 피검자가 입력한 문장들 중에서 검사자가 질의한 단어와 관련이 깊은 단어들을 추출하는 것이다. 이 방식을 통해 검사자가 질의한 단어에 대해 피검자가 어떠한 생각을 가지고 있는 지를 알 수 있다. 그러나 Word2vec을 쓰기에는 피검자로부터 입력되는 단어의 양이 지나치게 적어 학습할 데이터의 양이 부족하고 영어 단어를 데이터로 넣는 경우보다 결과가 좋지 않았다. 한국어는 영어와 달리 명사에 조사가 붙어 있는 경우가 많아서“여자”라는 단어와 관련이 있는 단어들을 추출하고 싶은 경우“여자는”이라는 단어를 입력한 경우와 “여자가”라는 단어를 입력했을 시에 결과가 달라진다. 이러한 문제점으로 인해 구글 API를 사용하였다. 


구글의 자연어 API에서 제공하는 메소드 중 엔터티 감정분석과 내용 분류는 한국어 지원이 안되므로 감정 분석 기능 만을 사용하였다. 감정분석은 텍스트 내에서 표현되는 전체적인 태도가 긍정적인지 부정적인지 판단한다. 이 때 감정은 숫자로 반환 되며 score와 magnitude 값으로 표현된다. 여기서 감정 score는 -1.0(부정적)에서 1.0(긍정적) 사이를 나타내며 텍스트의 전반적인 정서 성향을 나타낸다. magnitude는 주어진 텍스트 내에서 전반적인 감정의 강도를 나타내며 0.0부터 시작하여 무한대로 표현 된다. 텍스트 내의 각 감정 표현이 텍스트 magnitude에 반영되며, 따라서 긴 텍스트 블록일수록 값이 더 커지게 된다.





### 3.3 구현 결과


<img width = "600" src = "https://user-images.githubusercontent.com/28712478/52125910-3ee70c00-2671-11e9-8b7b-8e98f69c1ccc.png">
(표 1) 검사 분석 결과
 결과는 문항 별로 각 문장마다 감정 점수를 반환하고 하나의 문항에 들어오는 전체 문장의 감정 점수를 평균 낸 값과 magnitude를 반환한다. 표 1의 예시 문항1의 경우 내용이 전반적으로 부정적이기 때문에 전체 감정 점수가 -0.4인 반면 2번 문항의 내용은 전반적으로 평화롭고 긍정적이기 때문에 전체 감정 점수가 0.69로 분석이 되었다. 개별 문장으로 보면 문항2의 평균 점수는 0.69이지만 점수가 가장 높은 것은 4번째 문장으로‘아름다운 피아노 선율’,‘아름다운 아들의 모습’등 긍적적 단어가 서술 된 문장이고 점수가 가장 낮은 마지막 문장은 단순히 사실 정보를 나열한 것이다.
