import numpy as np
import sys


## CAI NI LA QUAN TRONG NHAT TRONG NI ROI

class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.
    Parameters
    ------------
    n_hidden : int (default: 30)
        So luong don vi an (mac dinh la 30 don vi).
    l2 : float (default: 0.)
        Gia tri Lambda cho L2-Regularization.
        Mac dinh khong co Regularization (l2 = 0)
    epochs : int (default: 100)
        So lan duyet qua toan bo du lieu train.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffle du lieu sau moi epoch de tranh hien tuong lap vong du lieu.
    minibatch_size : int (default: 1)
        So luong mau train cua mot minibatch.
    seed : int (default: None)
        Random seed khoi tao cac gia tri ngau nhien co dinh cho weights va dung de tron.
    Attributes
    -----------
    eval_ : dict
      Dictionary de luu tru cost, ti le training accuracy, va validation accuracy cho moi epoch trong qua trinh training.
    """

    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)   # Khoi tao randomState
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    # ONE_HOT ENCODING
    def _onehot(self, y, n_classes):
        """Ma hoa nhung label y thanh dang one-hot encoding
        Vi du y = [1, 2, 3], va n = 4 (n: la so luong class va cac phan tu trong y tinh chi so tu 0)
        => return = [[0. 1. 0. 0.]
                     [0. 0. 1. 0.]
                     [0. 0. 0. 1.]]
        Parameters
        ------------
        y : array, shape = [n_examples]
            Target values.
        n_classes : int
            Number of classes
        Returns
        -----------
        ## CAI CU CUA NGUOI TA
        onehot : array, shape = (n_examples, n_labels)

        Cach cua ho:
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T
        """

        ## CAI MINH VIET LAI
        onehot = np.zeros((y.shape[0], n_classes))
        for idx, val in enumerate(y.astype(int)):
            onehot[idx, val] = 1.
        return onehot

    # SIGMOID ACTIVATION: Ham dap ung cac nhu cau ben trong thoi
    def _sigmoid(self, z):
        """Tinh toan ham Logistic function (sigmoid)
        Tai sao nguoi ta lai keo gia tri cua thang z ve lai nguong -250, 250 ??
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Tinh toan gia tri cac weight cho buoc Forward Propagation
        Forward Propagation: Lan truyen xuoi
        Sử dụng những thứ có sẵn của thằng MLP: w_h, b_h, w_out, b_out

        """
        # Trong bai toan nay tac gia code 1 lop an thoi
        # STEP 1: Tinh gia tri net input cho lop an
        # (X: [n_examples, n_features]) dot w_h:[n_features, n_hidden]
        # -> hidden:[n_examples, n_hidden]
        # Bias thuc chat no la trong so lien ket cua thang a_0 trong moi lop voi lai nhung thang a trong lop tiep theo
        # Gia tri tai node cua thang a_0 la bang 1

        # z_h: [n_sample, n_hidden] + b_h: [n_hidden, ] Moi gia tri trong hang tang len luong bang voi hidden tai cot.
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        # Chuyen cac gia tri trong thang nay ve trong khoang 0 den 1
        # a_h: [n_sample, n_hidden]
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # a_h: [n_sample, n_hidden] dot w_out: [n_hidden, n_classlabels] -> [n_examples, n_classlabels]
        # z = wx + b

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        # Tra lai cac gia tri net input, activation of hidden va output layer
        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """Compute cost function.
        Parameters
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_examples, n_output_units]
            Activation of the output layer (forward propagation)
        Tinh toan cost function dua tren gia tri Encoding va Activation cua thang output
        Returns
        ---------
        cost : float
            Regularized cost
        """
        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)

        # Cost = Logistic Cost + L2_Term (Ridge regression)
        cost = np.sum(term1 - term2) + L2_term

        # If you are applying this cost function to other
        # datasets where activation
        # values maybe become more extreme (closer to zero or 1)
        # you may encounter "ZeroDivisionError"s due to numerical
        # instabilities in Python & NumPy for the current implementation.
        # I.e., the code tries to evaluate log(0), which is undefined.
        # To address this issue, you could add a small constant to the
        # activation values that are passed to the log function.
        #
        # For example:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)

        # Tạm dịch:
        # Nếu như ta áp dụng vào trong các dữ liệu khác mà giá trị activation có thể extreme
        # (tức là nó gần đến 0 hoặc 1) ta có thể gặp lỗi (Chia cho 0)
        # Ví dụ trong biểu thức này có thể dẫn đến log(0), giá trị undefined.
        # Để address vấn đề này, ta có thể thêm một vài hằng số nhỏ vào trong giá trị activation
        # rồi cho vào trong hàm log của mình
        # Ví dụ:
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)

        return cost

    # DU DOAN CLASS
    def predict(self, X):
        """Predict class labels
        Parameters
        -----------
        X : array, shape = [n_examples, n_features]
            Input layer with original features.
        Returns:
        ----------
        y_pred : array, shape = [n_examples]
            Predicted class labels.
        """
        # Tinh cac gia tri z_h, a_h, z_out, a_out bang cach feed forward vao trong mo hinh
        z_h, a_h, z_out, a_out = self._forward(X)
        # argmax(axis) neu ko co thi return vi tri thang max tren toan mang
        # neu axis=0 thi no duyet theo cot tai moi phan tu la chi so cua thang lon nhat cot, tra ra mang shape: [ncol]
        # neu axis=1 thi no duyet theo hang, tra lai mang []*n_sample voi moi thang la chi so cua thang lon nhat tai moi hang
        # Do thang z_out: [n_sample, n_label] nen minh lay argmax theo thang label
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    # HAM TRAIN DE TAO RA CAC GIA TRI w_h, b_h, w_out, b_out
    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.
        Gom co 2 bo du lieu Train va Validation
        Parameters
        -----------
        X_train : array, shape = [n_examples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_examples]
            Target class labels.
        X_valid : array, shape = [n_examples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_examples]
            Sample labels for validation during training
        Returns:
        ----------
        self
        """
        # So loai cac dau ra: dua vao label cua mau train ma dem
        n_output = np.unique(y_train).shape[0]  # number of class labels

        # So feature cua mau train, lay shape[1] cua thang train
        n_features = X_train.shape[1]  # 728

        ########################
        # Weight initialization
        ########################

        # Khoi tao weighs cho thang MLP: w_h, b_h, w_out, b_out
        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)  # [0.0] * n_hidden
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        # Dem so chu so cua thang epochs
        epoch_strlen = len(str(self.epochs))  # for progress formatting

        # Danh gia trong qua trinh train, com cac thong so: cost, train_accuracy, valid_accuracy
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        # Chuyen thang y_train tu dang [n_samples] sang dang [n_sample, n_labels] bang ham _onehot, n_output chinh la n_labels
        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            # Can mot bien indices de chi cho chi so cua thang X_train
            indices = np.arange(X_train.shape[0])

            # Tron index len sau moi epoch
            # Ta dung ham random: np.random.randomState(1).shuffle(index)
            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h)

                # [n_features, n_examples] dot [n_examples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # [n_hidden, n_examples] dot [n_examples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h  # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i + 1, self.epochs, cost,
                              train_acc * 100, valid_acc * 100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self


if __name__ == '__main__':
    a = NeuralNetMLP()
    print(a._onehot(np.array([1, 2, 3]), 4))
