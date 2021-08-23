#%%

# ALS Implementation - "Collaborative Filtering for 
#                       Implicit Feedback Datasets"

import numpy as np
import matplotlib.pyplot as plt

'''
    Define loss function:
        C        : confidence matrix
        P        : binary rating matrix
        xTy      : predict matrix
        X        : user latent matrix
        Y        : item latent matrix
        r_lambda : regularization lambda
    Total_loss = (confidence_level * predict loss) + regularization loss
'''
def loss_function(C, P, xTy, X, Y, r_lambda):
    predict_error = np.square(P-xTy)
    confidence_error = np.sum(C*predict_error)
    regularization = r_lambda*(np.sum(np.square(X))+np.sum(np.square(Y)))
    total_loss = confidence_error+regularization
    return np.sum(predict_error), confidence_error, regularization, total_loss

'''
    Define optimization functions for user and item:
        x[u]=(yT*C_{u}*Y+lambda*I)^-1 x yT*C_{u}*P_{u}
        y[u]=(xT*C_{u}*x+lambda*I)^-1 x xT*C_{u}*P_{u}
'''
def optimize_user(X, Y, C, P, nu, nf, r_lambda):
    # Y       : n(ni) x n(nf)
    # Y_Trans : n(nf) x n(ni)
    yT = np.transpose(Y)
    for u in range(nu):
        # Extract a diagonal or construct a diagonal array: np.diag()
        Cu = np.diag(C[u]) # nu x ni, u => ni x ni
        yT_Cu_y = np.matmul(np.matmul(yT, Cu), Y)

        # Make a identity matrix which is a square array with ones on the 
        # main diagonal: np.identity(nf) => nf X nf square itentity matrix
        lI = np.dot(r_lambda, np.identity(nf))
        yT_Cu_pu = np.matmul(np.matmul(yT, Cu), P[u])

        # Solve a linear matrix equation: np.linalg.solve()
        '''
        # example: 1*x0+2*x1=1, 3*x0+5*x1=2
            [1 2] [x0]   [1]
            [3 5] [x1] = [2]
            =>
            [x0]   [1 2]-1[1]
            [x1] = [3 4]  [2]
            
            >>> a = np.array([[1, 2], [3, 5]])
            >>> b = np.array([1, 2])
            >>> x = np.linalg.solve(a, b)
            >>> x
            array([-1.,  1.])
        '''
        X[u] = np.linalg.solve(yT_Cu_y + lI, yT_Cu_pu)

def optimize_item(X, Y, C, P, ni, nf, r_lambda):
    # X       : n(nu) x n(nf)
    # X_Trans : n(nf) x n(nu)
    xT = np.transpose(X)
    for i in range(ni):
        Ci = np.diag(C[:, i])
        xT_Ci_x = np.matmul(np.matmul(xT, Ci), X)
        lI = np.dot(r_lambda, np.identity(nf))
        xT_Ci_pi = np.matmul(np.matmul(xT, Ci), P[:, i])
        Y[i] = np.linalg.solve(xT_Ci_x + lI, xT_Ci_pi)

def train():
    predict_errors = []
    confidence_errors = []
    regularization_list = []
    total_losses = []

    for i in range(50):
        predict = np.matmul(X,np.transpose(Y))
        predict_error, confidence_error, regularization, total_loss = loss_function(C,P,predict,X,Y,r_lambda)

        predict_errors.append(predict_error)
        confidence_errors.append(confidence_error)
        regularization_list.append(regularization)
        total_losses.append(total_loss)

        print('----------------step %d----------------'%i)
        print("predict error: %f"%predict_error)
        print("confidence error: %f"%confidence_error)
        print("regularization: %f"%regularization)
        print("total loss: %f"%total_loss)

        optimize_user(X,Y,C,P,nu,nf,r_lambda)
        optimize_item(X,Y,C,P,ni,nf,r_lambda)

    predict = np.matmul(X,np.transpose(Y))
    print('final predict')
    print([predict])

    return predict_errors, confidence_errors, regularization_list, total_losses

def plot_losses(predict_errors, confidence_errors, regularization_list, total_losses):
    plt.subplots_adjust(wspace=100.0,hspace=20.0)
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    predict_error_line = fig.add_subplot(2,2,1)
    confidence_error_line = fig.add_subplot(2,2,2)
    regularization_error_line = fig.add_subplot(2,2,3)
    total_loss_line = fig.add_subplot(2,2,4)

    predict_error_line.set_title("Predict Error")
    predict_error_line.plot(predict_errors)

    confidence_error_line.set_title("Confidence Error")
    confidence_error_line.plot(confidence_errors)

    regularization_error_line.set_title("Regularization")
    regularization_error_line.plot(regularization_list)

    total_loss_line.set_title("Total Loss")
    total_loss_line.plot(total_losses)

    plt.show()

if __name__ == '__main__':
    '''
    1.  Initialize parameters:
            r_lambda : regularization parameter
            alpha    : confidence level
            nf       : dimension of latent vector of each user and item
    '''
    r_lambda = 30
    alpha = 40
    nf = 200

    '''
    2.  Initialize original rating data matrix (10 x 11):
        a r_{ui} can indicate the number of times 'u' purchased item 'i'
        or the time 'u' spent on webpage 'i'
            row(10) : num of users
            col(11) : num of items
    '''
    R = np.array([[0,0,0,4,4,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,0,1,0,4,0],
                  [0,3,4,0,3,0,0,2,2,0,0],
                  [0,5,5,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,5,0,0,5,0],
                  [0,0,4,0,0,0,0,0,0,0,5],
                  [0,0,0,0,0,4,0,0,0,0,4],
                  [0,0,0,0,0,0,5,0,0,5,0],
                  [0,0,0,3,0,0,0,0,4,5,0]])

    '''
    3.  Initialize user and item latent factor matrix with very small values:
            X : n(nu) x n(nf)
            : 10 x 200
            Y : n(ni) x n(nf)
            : 11 x 200
    '''
    nu = R.shape[0]
    ni = R.shape[1]
    X = np.random.rand(nu,nf)*0.01
    Y = np.random.rand(ni,nf)*0.01

    '''
    4.  Initialize Binary Matrix P:
        a set of binary variables p_{ui}, which indicates the preference
        of user 'u' to item 'i'
            p_{ui} = 1 if Rui > 0
            p_{ui} = 0 if Rui = 0
    '''
    P = np.copy(R)
    P[P>0]=1

    '''
    5.  Initialize Matrix C:
        a set of variables c_{ui}, which measure our confidence in
        observing p_{ui}
            c_{ui} = 1 + alpha*r_{ui}
    '''
    C = 1+alpha*R

    '''
    6.  Train the model
    '''
    predict_errors, confidence_errors, regularization_list, total_losses = train()

    '''
    7.  Show results 
    '''
    plot_losses(predict_errors,confidence_errors,regularization_list,total_losses)
# %%
