import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 체점을 위해 임의성을 사용하는 numpy 도구들의 결과가 일정하게 나오도록 해준다
np.random.seed(5)
RATING_DATA_PATH = './ratings.csv'  # 데이터 파일 경로 정의
# numpy 출력 옵션 설정
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def predict(Theta, X):
    """유저 취향과 상품 속성을 곱해서 예측 값을 계산하는 함수"""
    return Theta @ X


def cost(prediction, R):
    """행렬 인수분해 알고리즘의 손실을 계산해주는 함수"""
    return np.nansum((prediction - R)**2)


def initialize(R, num_features):
    """임의로 유저 취향과 상품 속성 행렬들을 만들어주는 함수"""
    num_users, num_items = R.shape
    
    Theta = np.random.rand(num_users, num_features)
    X = np.random.rand(num_features, num_items)
    
    return Theta, X


def gradient_descent(R, Theta, X, iteration, alpha, lambda_):
    """행렬 인수분해 경사 하강 함수"""
    num_user, num_items = R.shape
    num_features = len(X)
    costs = []
        
    for _ in range(iteration):
        prediction = predict(Theta, X)
        error = prediction - R
        costs.append(cost(prediction, R))
                          
        for i in range(num_user):
            for j in range(num_items):
                if not np.isnan(R[i][j]):
                    for k in range(num_features):
                        # 아래 코드를 채워 넣으세요.
                        Theta[i][k] -= alpha * (np.nansum(error[i, :]*X[k, :]) + lambda_*Theta[i][k])
                        X[k][j] -= alpha * (np.nansum(error[:, j]*Theta[:, k]) + lambda_*X[k][j])
                        
    return Theta, X, costs


#----------------------실행 코드----------------------
# 평점 데이터를 가지고 온다
ratings_df = pd.read_csv(RATING_DATA_PATH, index_col='user_id')

# 평점 데이터에 mean normalization을 적용한다
for row in ratings_df.values:
    row -= np.nanmean(row)
       
R = ratings_df.values
        
Theta, X = initialize(R, 5)  # 행렬들 초기화
Theta, X, costs = gradient_descent(R, Theta, X, 200, 0.001, 0.01)  # 경사 하강
    
# 손실이 줄어드는 걸 시각화 하는 코드 (디버깅에 도움이 됨)
plt.plot(costs)