import pandas as pd
import feature_gen as f
import model_eval as e

import logging
if __name__ == '__main__':

    # create logger with 'spam_application'
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    logger.info(__file__ + ' started')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 999)

    logger.info('loading data started')

    # Load Data
    logger.info('Loading train_transaction.csv')
    train_transaction = pd.read_csv('../data/train_transaction.csv')
    logger.info('train_transaction.csv loaded')

    logger.info('Loading train_identity.csv')
    train_identity = pd.read_csv('../data/train_identity.csv')
    logger.info('train_identity.csv loaded')


    train_transaction.set_index('TransactionID')

    # remove NANs
    train_transaction = train_transaction.fillna(0)


    # Join files
    logger.info('Joining files')
    df = train_transaction.join(train_identity.set_index('TransactionID')
                                , on='TransactionID'
                                , how='left')

    # Feature Generation

    # Encode Categorical
    to_encode = ['ProductCD', 'card4', 'card6',
                 'P_emaildomain', 'R_emaildomain',
                 'DeviceType', 'DeviceInfo'] + \
                ['M%i' % i for i in range(1, 10)]

    logger.info('Encoding categorical variables')
    features = f.OneHotEncode(df=df,
                              cols=to_encode)

    # Add continuous
    logger.info('Adding continuous variables')
    # features = features.join(df['TransactionAmt', 'V261'])
    features = pd.merge(features,
                        df[['TransactionAmt'] + ['V%i' % i for i in range(1, 339)]],
                        left_index=True,
                        right_index=True)

    # Add labels
    features = features.join(df['isFraud'])


    y_train, y_test, X_train, X_test = f.split_data(features=features,
                                                    split=0.2)

    ## Model
    logger.info('Training model')

    from sklearn import linear_model
    clf = linear_model.SGDClassifier(max_iter=1000,
                                     tol=1e-5,
                                     n_iter_no_change=10,
                                     random_state=1234,
                                     verbose=1)

    logger.info('Model parameters ' + str(clf))


    clf.fit(X=X_train,
            y=y_train)

    e.scores(clf, y_test, X_test)
