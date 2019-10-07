from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

def scores(clf, Y_test, X_test):
    y_true = Y_test
    y_pred = clf.predict(X_test)
    print('Confusuion matrix ' + str(confusion_matrix(y_true = y_true, y_pred = y_pred)))
    print('Accurary of %0.2f' % accuracy_score(y_true = y_true, y_pred = y_pred))
    print('Recall of %0.2f' % recall_score(y_true = y_true, y_pred = y_pred))
    print('Precision of %0.2f' % precision_score(y_true = y_true, y_pred = y_pred))