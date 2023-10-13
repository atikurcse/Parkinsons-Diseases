import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns

SVC_DIR = 'PaHaW/PaHaW_public'


class Subject(object):
    pass


def parse_corpus():
    dataset = pd.read_csv("PaHaW/PaHaW_files/corpus_PaHaW.csv", sep="\t", encoding="utf8")
    dataset['PD status'] = dataset['PD status'] == 'ON'
    dataset['PD status'] = dataset['PD status'].astype(int)
    return dataset


def generate_svc_path(subject_id, task):
    return "%s/%s/%s__%i_1.svc" % (SVC_DIR, subject_id, subject_id, task)


def displace(array):
    """Generates displacement vectors for columns in an array.

    Arg:
        array: np.array. Vectors to be displaced.
    """
    disp = np.zeros(shape=array.shape)
    disp[1:, :] = array[1:, :] - array[0:-1, :]
    return disp


def parse_svc(path):
    with open(path, 'r') as svc_file:
        samples = svc_file.readlines()

    # Extract the data
    data = []
    for sample in samples[1:]:
        values = [int(value) for value in sample.split()]
        data.append(values)
    array = np.array(data)
    n = array.shape[0]

    # position based velocity, acceleration, and jerk
    xy_vel = np.absolute(displace(array[:, 0:2]))
    xy_accel = displace(xy_vel)
    xy_jerk = displace(xy_accel)

    # magnitudes of previous measurements
    m_vel = np.linalg.norm(xy_vel, axis=1).reshape((n, 1))
    m_accel = np.linalg.norm(xy_accel, axis=1).reshape((n, 1))
    m_jerk = np.linalg.norm(xy_jerk, axis=1).reshape((n, 1))

    # rate of change of azimuth, altitude, and pressure
    aap_vel = np.absolute(displace(array[:, 4:6]))
    pr_cl = array[:, 6]
    pr_cl = displace(pr_cl.reshape((n, 1)))

    out = np.concatenate((array, xy_vel, xy_accel, xy_jerk, m_vel, m_accel,
                          m_jerk, aap_vel, pr_cl), axis=1)
    return out


def get_stat_values(task):
    writing_duration = np.max([task[:2]]) - np.min([task[:2]])
    p_pr_cnges = task[:, 18] > 0
    n_pr_cnges = task[:, 18] < 0
    on_surface = task[:, 3] == 1
    on_ait = task[:, 3] == 0
    summary_vect = np.array([
        np.sum(task[:, 13]) / writing_duration,  # on surface stroke speed
        np.std(task[:, 9]),  # horizontal velocity std
        np.std(task[on_surface, 14]),  # on surface magnitude acceleration std
        np.mean(np.absolute(task[on_surface, 16])),  # on surface azimuth velocity mean
        np.mean(np.absolute(task[on_surface, 17])),  # on surface altitude velocity mean
        np.std(np.absolute(task[on_surface, 17])),  # on surface altitude velocity std
        np.mean(task[on_surface, 6]),  # pressure mean
        np.std(task[on_surface, 6]),  # pressure std
        np.mean(task[on_surface, 4]),  # azimuth mean
        np.mean(task[on_surface, 5]),  # altitude mean
        np.std(task[on_surface, 4]),  # azimuth std
        np.std(task[on_surface, 5]),  # altitude std
        np.mean(task[p_pr_cnges, 18]),  # positive pressure changes mean
        np.std(task[p_pr_cnges, 18]),  # positive pressure changes std
        np.max(task[p_pr_cnges, 18]),  # positive pressure changes max
        np.mean(np.absolute(task[n_pr_cnges, 18])),  # negative pressure changes mean
        np.std(np.absolute(task[n_pr_cnges, 18])),  # negative pressure changes std
        np.min(task[n_pr_cnges, 18]),  # negative pressure changes max
        np.mean(np.absolute(task[:, 16])),  # azimuth velocity mean
        np.std(np.absolute(task[:, 16])),  # azimuth velocity std
        np.mean(np.absolute(task[:, 17])),  # altitude velocity mean
        np.std(np.absolute(task[:, 17])),  # altitude velocity std
        np.std(task[:, 13]),  # velocity magnitude std
        np.std(task[:, 14]),  # acceleration magnitude std
        np.std(task[:, 15]),  # jerk magnitude std
        np.mean(task[:, 13]),  # velocity magnitude mean
        np.mean(task[:, 14]),  # acceleration magnitude mean
        np.mean(task[:, 15]),  # jerk magnitude mean
        writing_duration,  # writing duration
        np.sum(task[:, 8]) / writing_duration  # speed

    ])

    return summary_vect




def summarize_features(subjects):
    x = []
    id=[]
    y = []

    for subject in subjects:
        sub_info = subject.info
        y.append(sub_info['PD status'])
        id.append(sub_info['ID'])
        temp = []
        for i in range(1, 9):
            try:
                temp.append(get_stat_values(subject.task[i]))
            except IOError:
                pass
        temp = np.array(temp)
        cd = [np.mean(temp[:, i]) for i in range(temp.shape[1])]
        x.append(cd)

    col = ['on_surface_stroke_speed', 'horizontal_vel_std', 'on_surface_magnitude_accel_std',
           'on_surface_azimuth_vel_mean', 'on_surface_altitude_vel_mean', 'on_surface_altitude_vel_std',
           'pr_mean', 'pr_std',  'azimuth_mean', 'altitude_mean', 'azimuth_std', 'altitude_std',
           'p_pre_change_mean', 'p_pre_change_std', 'p_pre_change_max', 'n_pre_change_mean',
           'n_pre_change_std', 'n_pre_change_max', 'azimuth_vel_mean', 'azimuth_vel_std',
           'altitude_vel_mean', 'altitude_vel_std', 'magn_vel_std', 'magn_accel_std', 'magn_jerk_std',
           'magn_vel_mean', 'magn_accel_mean', 'magn_jerk_mean', 'wri_duration', 'wri_speed']

    t1_data = pd.DataFrame(data=x, index=id, columns=col)
    y=np.array(y)
    n=y.shape[0]
    yn = pd.DataFrame(data=y, index=id, columns=['status'])
   
    return t1_data, y, yn


def extract_datasets():
    corpus = parse_corpus()
    inf = []
    subjects = []
    for i, row in enumerate(corpus.iterrows()):
        try:
            subject = Subject()
            inf = row[1]
            subject.info = row[1]
            subject.task = dict()
            subject_id = '%05d' % row[1].ID
            for tn in range(1, 9):

                svc_path = generate_svc_path(subject_id, tn)
                task_data = parse_svc(svc_path)
                subject.task[tn] = task_data
        except IOError:
            print("Error id: ", row[1].ID, " task :", tn)
            continue

        subjects.append(subject)
    return subjects


def feature_selection(x, y, i):
    """
     test=SelectKBest(score_func=f_classif, k=5)
     fit=test.fit(x,y)
     np.set_printoptions(precision=3)
     print(fit.scores_)
    """
    selector = SelectKBest(score_func=f_classif, k=i)
    selector.fit(x, y)
    cols = selector.get_support(indices=True)
    n_x = x.iloc[:, cols]
    return n_x


def create_model(x, y):
    X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)
    models = []
    models.append(('Logistic Regression(LR) ', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('Linear Discriminant Analysis (LDA)', LinearDiscriminantAnalysis()))
    models.append(('K-Nearest Neighbors (KNN)', KNeighborsClassifier()))
    models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))
    models.append(('Naïve Bayes (NB)', GaussianNB()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print("model :", name)
        print('  Accuracy: %.3f' % accuracy_score(Y_validation, predictions))
        print(' Precision: %.3f' % precision_score(Y_validation, predictions))
        print('    Recall: %.3f' % recall_score(Y_validation, predictions))     	
        print('  F1 Score: %.3f' % f1_score(Y_validation, predictions))
        print("-----------------\n")

    


if __name__ == '__main__':
    print("Extracting data")
    subjects = extract_datasets()
    x, y, yn = summarize_features(subjects)
    n_x = feature_selection(x, y, 7)
    print("\n---------------------------------------\n")
    print("After feature selection model performance\n")
    print("-----------------------------------------\n")
    create_model(n_x, y)
    print("-----------------------------------------\n")
    print("Without feature selection model performance\n")
    print("-----------------------------------------\n")
    create_model(x, y)
    
    df = pd.concat([n_x, yn], axis=1, join="inner")
    
    features = df.columns.difference(['status'])
    group_0_df = df[df['status'] == 0]
    group_1_df = df[df['status'] == 1]

    # Define colors for the two groups
    group_0_color = '#008000'
    group_1_color = '#990033'

    # Create a 2x4 grid of subplots with adjusted figure size
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))

    # Generate box plots for each feature for both groups
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2

        # Plot group 0 data with one color
        axes[row, col].boxplot(group_0_df[feature], positions=[1], patch_artist=True, boxprops={'facecolor': group_0_color})
        # Plot group 1 data with another color
        axes[row, col].boxplot(group_1_df[feature], positions=[2], patch_artist=True, boxprops={'facecolor': group_1_color})

        # Set the feature name as the subplot title
        axes[row, col].set_title(feature, fontsize=12, color='green')

    # Set x-axis labels
    for ax in axes.flat:
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Healthy', 'PD'])

    # Adjust subplot spacing
    plt.tight_layout()

    # Add a common title for the entire figure
    
    plt.show()