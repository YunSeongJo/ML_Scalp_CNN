# ML_Scalp_CNN (두피 질환 예측 모델 생성, 정확도: 77%)


https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=216
Aihub에서 제공하는 두피 이미지 세트 사용

이미지: 640 X 480 size

8가지의 클래스중 정확하게 구별하기 위해 4가지 클래스로 줄임(비듬,탈모,정상,염증성) 그리고 중증의 이미지만 사용
Train 개수: 비듬(2200여개), 정상(530여개), 염증(332개), 탈모(800개)



# ImageArgumentation으로 데이터 증강

ImageArgumentation 파일 참고

데이터 불균형을 해결하기 위해 각 클래스마다 2000여개의 데이터를 맞춤

생성 방식 : 기존 이미지의 상하 반전, 좌우 반전, 상하 좌우 반전으로 3배를 더해 총 4배까지 늘림. (비듬은 원래 이미지 개수가 2200개 이기 때문에 증강하지 않았음)



# 전처리

랜덤 시드를 고정시키고 전체 훈련데이터 8228개를 8:2로 나누어서 검증데이터를 생성

훈련 데이터, 검증 데이터를 가져와서 rescale로 정규화 시킨 후, target_size를 리사이징 시킨다음 batch_size 설정하여 One-Hot Encoding을 classmode로 설정

테스트 데이터는 새로운 데이터 40개만 테스트


# 모델 구성

Sequential 모델을 사용

기본적인 CNN모델이며 Conv2D와 MaxPooling2D, Flatten, Dense를 사용하여 구성
자세한 설명은 나중에...


따라서 모델을 학습시키고 평가를 하면 

![최종 accuracy 그래프](https://github.com/YunSeongJo/ML_Scalp_CNN/assets/50141553/994747c3-e166-4e00-a9b5-737ebd0b54e5)
![최종 loss 그래프](https://github.com/YunSeongJo/ML_Scalp_CNN/assets/50141553/a5ab9c06-6dfc-4457-890f-376749458b2b)

최종 accuracy와 loss그래프를 생성한다.

train은 학습이 잘되지만 loss는 학습이 잘 안되는 것을 볼 수 있다. 아마 기존 이미지의 특성 추출이나 메타 데이터 등이 필요할 것으로 예상..

마지막으로 테스트 40장을 모델에 돌려보면 40%정도로 맞추는 것을 확인 할 수 있다.

일반화가 안되긴 하지만 어느정도 vaildation 데이터 셋의 evaluate는 77%나오는 것을 보아하니 개선의 여지는 있다.

자세한 코드 내용은 ML_Final_Hair_scalp.py를 확인하면 된다.
