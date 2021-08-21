# Machine Learning 
## Index
* [Machine Learning](#machine-learning)

---

## Machine Learning
> "Field of study that gives computers the ability to learn without being explicitly programmed" Arthur Samuel (1959)

* 개발자가 explicit하게 작성한 프로그램은 한계가 있음:
    * Spam filter: many rules
    * Automatic driving: TOO many rules

* 어떠한 프로그램(ML)이 explicit하게 작성하지 않고도 주어진 <b>데이터</b>를 통해 <b>학습</b>하여 유의미한 <b>결과</b>를 도출

* Machine Learning은 크게 두 가지로 분류할 수 있음:
    * Label 유무에 따라 구분됨
    * Supervised learning (regression) : learning with labeled examples
    * Unsupervised learning (clustering) : un-labeded data

### Supervised Learning
* Types of supervised learning
    * Predicting final exam score based on time spent (<b>regression</b>)
    * Pass/non-pass based on time spent (<b>binary classification</b>)
    * Letter grade (A, B, C, E, and F) based on time spent (<b>multi-label classification</b>)

#### Linear Regression (in supervised learning)
* 1차 함수를 사용하여 data를 어떻게 fitting할 것인가?
    * <i>y</i> = <i>Wx</i> + <i>b</i>라는 model이 주어졌을 때, <i>W</i>와 <i>b</i>를 어떻게 최적화할 것인가?
* Linear regression의 사용 목적은 임의의 data를 model에 넣었을 때 결과를 prediction하기 위함

1. Hypothesis

    <img src = "./img/lr_hypo.png" width="40%">

2. Cost/loss function

    ![cost-function](./img/lr_cf.png)
    * 비용이 클수록(함수의 결과가 클수록) 평균과 차이가 크다는 뜻
    * 평균으로 가기 위한 비용이 많이 발생한다는 뉘앙스

3. Goal (Algorithm): Gradient Descent Algorithm
* 동작 과정: 
    1. Make convex function(model을 제곱)

        ![convex-function](./img/lr_convex.png)
        * Linear regression에서 사용하는 cost function은 convex하게 모양이 잘 나옴
        * 그러나 cost function이 복잡해지면 cost function을 convex하게 만들기 어려움
            * 'local minima' problem에 빠질 수 있음
            * 해당 문제를 해결하는 방법은 오늘날에도 활발히 연구되고 있는 분야
    
    2. <i>W</i>, <i>b</i>에 대해 편미분
    3. <i>W</i>, <i>b</i> 갱신(update)

    ![gradient-descent1](./img/lr_gd1.png)

    ![gradient-descent2](./img/lr_gd2.png)

    ![gradient-descent3](./img/lr_gd3.png)

    ![gradient-descent4](./img/lr_gd4.png)





