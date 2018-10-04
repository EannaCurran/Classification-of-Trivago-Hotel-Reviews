'''

Summer Project 2017 By Eanna Curran
Machine Learning to Predict if a  Trivago Review is Helpful or Unhelpful

'''

import re
import pandas as pd

from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier


# Divides Raw Data And Appends it to a Dictionary
class DivideData:

    def __init__(self, splitData):
        self.splitData = splitData

    def ExtractData(self, splitData):
        for n in splitData:

            meanHotelRating = (n[8])
            dataSet['Mean_Hotel_Rating'].append(meanHotelRating)

            numberHotelRating = (n[11])
            dataSet['Number_Of_Hotel_Ratings'].append(numberHotelRating)

            reviewHistogram = (n[14])
            dataSet['Review_Histogram'].append(reviewHistogram)

            dateOfReview = (n[20])
            dataSet['Date_Of_Review'].append(dateOfReview)

            reviewText = (n[26])
            reviewText = re.sub('<[^>]*>', '', reviewText)
            dataSet['Review_Text'].append(reviewText)
            reviewTotalChar = 0
            for char in reviewText:
                reviewTotalChar += 1
            dataSet['Review_Total_Characters'].append(reviewTotalChar)

            rating = (n[29])
            dataSet['Rating'].append(rating)

            helpfulness = (n[32])
            dataSet['Helpfulness'].append(helpfulness)

            value = (n[35])
            dataSet['Value_Rating'].append(value)

            room = (n[38])
            dataSet['Room_Rating'].append(room)

            location = (n[41])
            dataSet['Location_Rating'].append(location)

            cleanliness = (n[44])
            dataSet['Cleanliness_Rating'].append(cleanliness)

            service = (n[50])
            dataSet['Service_Rating'].append(service)

            dateOfStay = (n[56])
            dataSet['Date_Of_Stay'].append(dateOfStay)

            visitWasFor = (n[59])
            dataSet['Visit_Was_For'].append(visitWasFor)

            memberSince = (n[68])
            dataSet['Member_Since'].append(memberSince)

            numberOfQuestions = (n[77])
            dataSet['Number_Of_Questions'].append(numberOfQuestions)


# Removes Reviews With Invalied/Null Data or Reviews With Less Then 5 Feedback on it
class CleanData:

    def __init__(self, dataframe):
        self.dataframe = dataframe

# Replaces 'of' in Review Ratings With '/'
    def HelpfulnessFix(self, dataframe):
        new = []
        for n in dataframe['Helpfulness']:
            n = n.replace(' of ', '/')
            new.append(n)
        df['Helpfulness'] = new
        df.index = range(0, 78123)

# Removes Reviews With Ratings That Have Less Then 5 Feedback on it
    def RemoveUnhelpful(self, dataframe):
        df_remove = []
        global df
        for n in dataframe['Helpfulness']:
            if len(n) == 3:
                if int(n[-1]) < 5:
                    df_remove.append(n)
        df = df[~df['Helpfulness'].isin(df_remove)]

# Removes Reviews With Invalid Helpfulness Data Inputs
    def InvalidHelpfulness(self, dataframe):
        Invalid_Data = []
        global df
        for n in dataframe['Helpfulness']:
            try:
                int((n.rsplit('/', 1)[0]))
                int((n.rsplit('/', 1)[-1]))
            except ValueError:
                Invalid_Data.append(n)
        df = df[~df['Helpfulness'].isin(Invalid_Data)]

# Removes Reviews With Invalid Rating Data Inputs
    def InvalidRating(self, dataframe):
        Invalid_Rating = []
        global df
        for n in dataframe['Rating']:
            try:
                n = int(n)
            except ValueError:
                Invalid_Rating.append(n)
        df = df[~df['Rating'].isin(Invalid_Rating)]

# Removes Review With no Review Text in it
    def ZeroReview(self, dataframe):
        Zero_Review = []
        global df
        for n in dataframe['Review_Total_Characters']:
            if int(n) == 0:
                Zero_Review.append(n)
        df = df[~df['Review_Total_Characters'].isin(Zero_Review)]

# Creates a New Dataset For if The Review Helpfulness Score Was Over 60%
    def HelpfulList(self, dataframe):
        Helpfulness_List = []
        Name = []
        for n in dataframe['Helpfulness']:
            helpful_rating = int((n.rsplit('/', 1)[0]))
            total_rating = int((n.rsplit('/', 1)[-1]))
            if helpful_rating / total_rating >= 0.6:
                Helpfulness_List.append(1)
            elif helpful_rating / total_rating < 0.6:
                Helpfulness_List.append(0)
        df['Targets'] = df['Helpfulness']
        df['Targets'] = Helpfulness_List
        for n in df['Targets']:
            if int(n) == 0:
                Name.append('Unhelpful')
            elif int(n) == 1:
                Name.append('Helpful')
        df['Name'] = df['Helpfulness']
        df['Name'] = Name

# Removes Reviews With Null Inputs For Dates
    def NullDates(self, dataframe):
        NullStay = []
        NullMember = []
        global df
        for n in dataframe['Date_Of_Stay']:
            if n == 'null':
                NullStay.append(n)
        for n in dataframe['Member_Since']:
            if n == 'null':
                NullMember.append(n)
        df = df[~df['Date_Of_Stay'].isin(NullStay)]
        df = df[~df['Member_Since'].isin(NullMember)]


# Creates New Features For Machine Learning Algorithim to Classify With
class Features:

    def __init__(self, dataframe):
        self.dataframe = dataframe

# Counts The Number of Extra Ratinngs in The Review
    def TotalExtraRatings(self, dataframe):
        Extra_Rating = []
        a1 = dataframe['Cleanliness_Rating']
        b1 = dataframe['Location_Rating']
        c1 = dataframe['Room_Rating']
        d1 = dataframe['Service_Rating']
        e1 = dataframe['Value_Rating']
        for a, b, c, d, e in zip(a1, b1, c1, d1, e1):
            total = 0
            if a == 'null':
                pass
            else:
                total += 1
            if b == 'null':
                pass
            else:
                total += 1
            if c == 'null':
                pass
            else:
                total += 1
            if d == 'null':
                pass
            else:
                total += 1
            if e == 'null':
                pass
            else:
                total += 1
            Extra_Rating.append(total)
        df['Extra_Ratings'] = df['Helpfulness']
        df['Extra_Ratings'] = Extra_Rating

# Calculates The Number of Times The User Has/Hasnt Given The Rating Before
    def DifferenceOfReviews(self, dataframe):
        SameRating = []
        DifferenceRating = []
        Checker = []
        Histogram = dataframe['Review_Histogram']
        Rating = dataframe['Rating']
        NumRatings = dataframe['Number_Of_Hotel_Ratings']
        for n in Histogram:
            Checker.append(n.split(','))
        for n, m, o in zip(Checker, Rating, NumRatings):
            if int(m) == 5:
                Same = int(n[0])
                SameRating.append(Same)
                Difference = int(o) - Same
                DifferenceRating.append(Difference)
            elif int(m) == 4:
                Same = int(n[1])
                SameRating.append(Same)
                Difference = int(o) - Same
                DifferenceRating.append(Difference)
            elif int(m) == 3:
                Same = int(n[2])
                SameRating.append(Same)
                Difference = int(o) - Same
                DifferenceRating.append(Difference)
            elif int(m) == 2:
                Same = int(n[3])
                SameRating.append(Same)
                Difference = int(o) - Same
                DifferenceRating.append(Difference)
            elif int(m) == 1:
                Same = int(n[4])
                SameRating.append(Same)
                Difference = int(o) - Same
                DifferenceRating.append(Difference)
        df['Same_Rating'] = df['Helpfulness']
        df['Same_Rating'] = SameRating
        df['Different_Rating'] = df['Helpfulness']
        df['Different_Rating'] = DifferenceRating

# Calculates The Lenght of Time Between Becoming a Member of Trivago And Writing The Review
    def MemberToReviewDate(self, dataframe):
        NumToReview = []
        for n, m in zip(df['Member_Since'], df['Date_Of_Review']):
            DayMember = int(n[-8:-6])
            YearMember = int(n[-4:])
            MonthMember = n[0:3]
            MonthMember = MonthNumber(MonthMember)
            DateMember = YearMember * 10000 + MonthMember * 100 + DayMember

            DayReview = int(m[-8:-6])
            YearReview = int(m[-4:])
            MonthReview = m[0:3]
            MonthReview = MonthNumber(MonthReview)
            DateReview = YearReview * 10000 + MonthReview * 100 + DayReview

            DifferenceInDates = DateReview - DateMember
            NumToReview.append(DifferenceInDates)
        df['Date_To_Review'] = df['Helpfulness']
        df['Date_To_Review'] = NumToReview

# Calcutes The Length of Time Between The Visit And Writing The Review
    def AgeOfReview(self, dataframe):
        Age = []
        for n, m in zip(dataframe['Date_Of_Review'], dataframe['Date_Of_Stay']):
            MonthReview = n[0:3]
            YearReview = int(n[-4:])
            MonthReview = MonthNumber(MonthReview)
            DateReview = YearReview * 10000 + MonthReview * 100

            MonthStay = m[0:3]
            YearStay = int(m[-4:])
            MonthStay = MonthNumber(MonthStay)
            DateStay = YearStay * 10000 + MonthStay * 100

            DifferenceInDates = DateReview - DateStay
            Age.append(DifferenceInDates)
        df['Age_Of_Review'] = df['Helpfulness']
        df['Age_Of_Review'] = Age

# Calculates The Differce Between The Rating Given And The Sentiment of The Review Text
    def Sentiment(self, dataframe):
        PolarityDifference = []
        for n, m in zip(dataframe['Review_Text'], dataframe['Rating']):
            Text = TextBlob(n)
            Polarity = Text.sentiment.polarity
            Rating = int(m) / 5
            Result = Rating - Polarity
            PolarityDifference.append(Result)
        df['Polarity_Difference'] = df['Helpfulness']
        df['Polarity_Difference'] = PolarityDifference


def Group(seq, sep):
    newList = []
    for element in seq:
        if element == sep:
            yield newList
            newList = []
        newList.append(element)
    yield newList


def MonthNumber(month):
    if month == 'jan':
        return 1
    elif month == 'feb':
        return 2
    elif month == 'mar':
        return 3
    elif month == 'apr':
        return 4
    elif month == 'may':
        return 5
    elif month == 'jun':
        return 6
    elif month == 'jul':
        return 7
    elif month == 'aug':
        return 8
    elif month == 'sep':
        return 9
    elif month == 'oct':
        return 10
    elif month == 'nov':
        return 11
    elif month == 'dec':
        return 12


def Results(clf, targets):
    TruePositive = 0
    FalsePositive = 0
    TrueNegative = 0
    FalseNegative = 0
    for n, m in zip(clf, targets):
        if n == int(m):
            if n == 0:
                TrueNegative += 1
            elif n == 1:
                TruePositive += 1
        elif n != int(m):
            if n == 0:
                FalseNegative += 1
            elif n == 1:
                FalsePositive += 1
    print('True Negative: {}'.format(TrueNegative))
    print('True Positive: {}'.format(TruePositive))
    print('False Negative: {}'.format(FalseNegative))
    print('False Positive: {}'.format(FalsePositive))
    TotalTrue = TruePositive + TrueNegative
    Total = TruePositive + TrueNegative + FalsePositive + FalseNegative
    Accuracy = TotalTrue / Total
    print('Accuracy: {}'.format(Accuracy))


def TestData(df):
    df_test = df
    df_test2 = df
    test_1 = []
    test_0 = []
    count_1 = 0
    count_0 = 0
    for n in df_test['Targets']:
        if n == 1 and count_1 < 1250:
            test_1.append(n)
            count_1 += 1
        elif n == 0 and count_0 < 1250:
            test_0.append(n)
            count_0 += 1
    df_test = df[df_test['Targets'].isin(test_1)]
    df_test2 = df[df_test2['Targets'].isin(test_0)]
    df_test = df_test.head(1000)
    df_test2 = df_test2.head(1000)
    train_data = df_test.append(df_test2)
    return train_data


def RemoveTestData():
    train_index = []
    for n in train_data.index:
        train_index.append(n)
    sorted(train_index, key=int)
    for n in train_index:
        df.drop(n, inplace=True)


dataSet = {'Mean_Hotel_Rating': [],
           'Number_Of_Hotel_Ratings': [],
           'Review_Histogram': [],
           'Date_Of_Review': [],
           'Review_Total_Characters': [],
           'Review_Text': [],
           'Rating': [],
           'Helpfulness': [],
           'Value_Rating': [],
           'Room_Rating': [],
           'Location_Rating': [],
           'Cleanliness_Rating': [],
           'Service_Rating': [],
           'Date_Of_Stay': [],
           'Visit_Was_For': [],
           'Member_Since': [],
           'Number_Of_Questions': []}

file = 'chicago.txt'
f = open(file, encoding='ISO-8859-1')
text = f.read().splitlines()

splitData = list(Group(text, '<review>'))
splitData.pop(0)

result = DivideData(splitData)
result.ExtractData(splitData)

df = pd.DataFrame(dataSet)


data = CleanData(df)
data.HelpfulnessFix(df)
data.RemoveUnhelpful(df)
data.InvalidHelpfulness(df)
data.InvalidRating(df)
data.ZeroReview(df)
data.HelpfulList(df)
data.NullDates(df)

Features = Features(df)
Features.TotalExtraRatings(df)
Features.DifferenceOfReviews(df)
Features.MemberToReviewDate(df)
Features.AgeOfReview(df)
Features.Sentiment(df)

train_data = TestData(df)
RemoveTestData()

train = train_data[['Rating', 'Review_Total_Characters',
                    'Extra_Ratings',
                    'Number_Of_Hotel_Ratings', 'Different_Rating',
                    'Polarity_Difference', 'Targets']]

data = df[['Rating', 'Review_Total_Characters',
           'Extra_Ratings',
           'Number_Of_Hotel_Ratings', 'Different_Rating',
           'Polarity_Difference']]

features = list(train.columns[:-1])
targets = ['Helpful', 'Unhelpful']
x = train[features]
y = train['Targets']
clf = RandomForestClassifier()
clf = clf.fit(x, y)

clf = clf.predict(data)

Results(clf, df['Targets'])
