{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f5680d2c",
   "metadata": {
    "id": "f5680d2c"
   },
   "source": [
    "# 1. Preprocessing\n",
    "\n",
    "## 1-a. Import Data and libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "38d9c9b8",
   "metadata": {
    "id": "38d9c9b8"
   },
   "outputs": [],
   "source": [
    "#Loading library\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import re\n",
    "import time\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "85cde8ea",
   "metadata": {
    "id": "85cde8ea",
    "outputId": "27251a0f-f7d2-425a-d34d-6a3721d95d7b"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>category</th>\n",
       "      <th>viewCount</th>\n",
       "      <th>likeCount</th>\n",
       "      <th>commentCount</th>\n",
       "      <th>duration</th>\n",
       "      <th>made_for_kids</th>\n",
       "      <th>viewPerComment</th>\n",
       "      <th>tagsCount</th>\n",
       "      <th>year</th>\n",
       "      <th>weekday</th>\n",
       "      <th>month</th>\n",
       "      <th>time_of_day</th>\n",
       "      <th>title_length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>39271</td>\n",
       "      <td>2538</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>165.401103</td>\n",
       "      <td>8</td>\n",
       "      <td>2023</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>156694</td>\n",
       "      <td>6633</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>364.847128</td>\n",
       "      <td>6</td>\n",
       "      <td>2023</td>\n",
       "      <td>6</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>78836</td>\n",
       "      <td>2315</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>742.543413</td>\n",
       "      <td>3</td>\n",
       "      <td>2023</td>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>34</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>119458</td>\n",
       "      <td>5293</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>390.751181</td>\n",
       "      <td>9</td>\n",
       "      <td>2023</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>113299</td>\n",
       "      <td>4649</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>440.775436</td>\n",
       "      <td>9</td>\n",
       "      <td>2023</td>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   category  viewCount  likeCount  commentCount  duration  made_for_kids   \n",
       "0         5          2      39271          2538         4              0  \\\n",
       "1         5          4     156694          6633         3              0   \n",
       "2         5          4      78836          2315         5              0   \n",
       "3         5          4     119458          5293         2              0   \n",
       "4         5          4     113299          4649         2              0   \n",
       "\n",
       "   viewPerComment  tagsCount  year  weekday  month  time_of_day  title_length  \n",
       "0      165.401103          8  2023        0      5            0            18  \n",
       "1      364.847128          6  2023        6      5            0            35  \n",
       "2      742.543413          3  2023        5      5            1            34  \n",
       "3      390.751181          9  2023        4      5            0            30  \n",
       "4      440.775436          9  2023        3      5            1            16  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"cleaned_data.csv\",index_col=0)\n",
    "df = df.drop('time',axis=1)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "36ffa9ed",
   "metadata": {
    "id": "36ffa9ed",
    "outputId": "6b996ece-148f-4bfb-910d-c28b913b3b5f"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>category</th>\n",
       "      <th>viewCount</th>\n",
       "      <th>likeCount</th>\n",
       "      <th>commentCount</th>\n",
       "      <th>duration</th>\n",
       "      <th>made_for_kids</th>\n",
       "      <th>viewPerComment</th>\n",
       "      <th>tagsCount</th>\n",
       "      <th>year</th>\n",
       "      <th>weekday</th>\n",
       "      <th>month</th>\n",
       "      <th>time_of_day</th>\n",
       "      <th>title_length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>3.462100e+04</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "      <td>34621.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>6.226972</td>\n",
       "      <td>3.089310</td>\n",
       "      <td>7.358566e+04</td>\n",
       "      <td>4437.750296</td>\n",
       "      <td>1.826926</td>\n",
       "      <td>0.000289</td>\n",
       "      <td>883.064358</td>\n",
       "      <td>15.499928</td>\n",
       "      <td>2017.403223</td>\n",
       "      <td>2.958378</td>\n",
       "      <td>6.479160</td>\n",
       "      <td>0.827821</td>\n",
       "      <td>50.488721</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.528896</td>\n",
       "      <td>1.501606</td>\n",
       "      <td>2.058486e+05</td>\n",
       "      <td>10732.664863</td>\n",
       "      <td>1.433032</td>\n",
       "      <td>0.016993</td>\n",
       "      <td>1395.020067</td>\n",
       "      <td>11.285734</td>\n",
       "      <td>3.458414</td>\n",
       "      <td>1.949366</td>\n",
       "      <td>3.455963</td>\n",
       "      <td>0.611762</td>\n",
       "      <td>22.123522</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2008.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.579000e+03</td>\n",
       "      <td>337.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>313.686341</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>2015.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>35.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.400400e+04</td>\n",
       "      <td>1487.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>540.695203</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>2018.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>48.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>7.497000e+04</td>\n",
       "      <td>4660.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1011.403274</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>2020.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>64.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>14.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>1.219854e+07</td>\n",
       "      <td>617657.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>89817.454545</td>\n",
       "      <td>83.000000</td>\n",
       "      <td>2023.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>101.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           category     viewCount     likeCount   commentCount      duration   \n",
       "count  34621.000000  34621.000000  3.462100e+04   34621.000000  34621.000000  \\\n",
       "mean       6.226972      3.089310  7.358566e+04    4437.750296      1.826926   \n",
       "std        3.528896      1.501606  2.058486e+05   10732.664863      1.433032   \n",
       "min        0.000000      0.000000  0.000000e+00       0.000000      0.000000   \n",
       "25%        5.000000      2.000000  4.579000e+03     337.000000      1.000000   \n",
       "50%        5.000000      3.000000  2.400400e+04    1487.000000      2.000000   \n",
       "75%        7.000000      4.000000  7.497000e+04    4660.000000      2.000000   \n",
       "max       14.000000      9.000000  1.219854e+07  617657.000000      5.000000   \n",
       "\n",
       "       made_for_kids  viewPerComment     tagsCount          year   \n",
       "count   34621.000000    34621.000000  34621.000000  34621.000000  \\\n",
       "mean        0.000289      883.064358     15.499928   2017.403223   \n",
       "std         0.016993     1395.020067     11.285734      3.458414   \n",
       "min         0.000000        0.000000      0.000000   2008.000000   \n",
       "25%         0.000000      313.686341      7.000000   2015.000000   \n",
       "50%         0.000000      540.695203     14.000000   2018.000000   \n",
       "75%         0.000000     1011.403274     23.000000   2020.000000   \n",
       "max         1.000000    89817.454545     83.000000   2023.000000   \n",
       "\n",
       "            weekday         month   time_of_day  title_length  \n",
       "count  34621.000000  34621.000000  34621.000000  34621.000000  \n",
       "mean       2.958378      6.479160      0.827821     50.488721  \n",
       "std        1.949366      3.455963      0.611762     22.123522  \n",
       "min        0.000000      1.000000      0.000000      1.000000  \n",
       "25%        1.000000      3.000000      0.000000     35.000000  \n",
       "50%        3.000000      6.000000      1.000000     48.000000  \n",
       "75%        5.000000      9.000000      1.000000     64.000000  \n",
       "max        6.000000     12.000000      2.000000    101.000000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e15780db",
   "metadata": {
    "id": "e15780db"
   },
   "source": [
    "#### KeyTable for references:\n",
    "\n",
    "1. Category\n",
    "![image.png](attachment:image.png)\n",
    "\n",
    "2. Duration\n",
    "\n",
    "![image-3.png](attachment:image-3.png)\n",
    "\n",
    "3. View Counts (Response Variable)\n",
    "\n",
    "![image-2.png](attachment:image-2.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d15566a",
   "metadata": {
    "id": "1d15566a"
   },
   "source": [
    "## 1-b. Train, test split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "74e21da7",
   "metadata": {
    "id": "74e21da7"
   },
   "outputs": [],
   "source": [
    "#split the data into train and test set\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "np.random.seed(1111)\n",
    "raw_train, raw_test = train_test_split(df, test_size = 0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2e116f02",
   "metadata": {
    "id": "2e116f02"
   },
   "outputs": [],
   "source": [
    "X = df.drop(['viewCount'], axis = 1)\n",
    "y = df['viewCount']\n",
    "\n",
    "X_train, X_test, y_train,  y_test  = train_test_split(X, y, test_size = 0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "daf2b09a",
   "metadata": {
    "id": "daf2b09a",
    "outputId": "cec4012c-8c82-4f8b-a9f8-c107d6958691"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(27696,)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b1484f53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>category</th>\n",
       "      <th>likeCount</th>\n",
       "      <th>commentCount</th>\n",
       "      <th>duration</th>\n",
       "      <th>made_for_kids</th>\n",
       "      <th>viewPerComment</th>\n",
       "      <th>tagsCount</th>\n",
       "      <th>year</th>\n",
       "      <th>weekday</th>\n",
       "      <th>month</th>\n",
       "      <th>time_of_day</th>\n",
       "      <th>title_length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>30735</th>\n",
       "      <td>3</td>\n",
       "      <td>1979</td>\n",
       "      <td>817</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>151.157895</td>\n",
       "      <td>27</td>\n",
       "      <td>2020</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>96</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9957</th>\n",
       "      <td>5</td>\n",
       "      <td>7768</td>\n",
       "      <td>268</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1254.167910</td>\n",
       "      <td>20</td>\n",
       "      <td>2017</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>68</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25156</th>\n",
       "      <td>1</td>\n",
       "      <td>36242</td>\n",
       "      <td>2948</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>813.363976</td>\n",
       "      <td>38</td>\n",
       "      <td>2018</td>\n",
       "      <td>3</td>\n",
       "      <td>9</td>\n",
       "      <td>2</td>\n",
       "      <td>62</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17713</th>\n",
       "      <td>12</td>\n",
       "      <td>1430</td>\n",
       "      <td>381</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>50.144357</td>\n",
       "      <td>0</td>\n",
       "      <td>2013</td>\n",
       "      <td>2</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>36</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16289</th>\n",
       "      <td>12</td>\n",
       "      <td>34432</td>\n",
       "      <td>1915</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>874.626110</td>\n",
       "      <td>13</td>\n",
       "      <td>2016</td>\n",
       "      <td>5</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "      <td>45</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2374</th>\n",
       "      <td>5</td>\n",
       "      <td>84648</td>\n",
       "      <td>8158</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>400.011890</td>\n",
       "      <td>24</td>\n",
       "      <td>2016</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32682</th>\n",
       "      <td>3</td>\n",
       "      <td>489</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>683.384615</td>\n",
       "      <td>26</td>\n",
       "      <td>2017</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6066</th>\n",
       "      <td>3</td>\n",
       "      <td>295356</td>\n",
       "      <td>8281</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1289.378819</td>\n",
       "      <td>6</td>\n",
       "      <td>2018</td>\n",
       "      <td>6</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "      <td>39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25250</th>\n",
       "      <td>1</td>\n",
       "      <td>35678</td>\n",
       "      <td>3664</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>802.075328</td>\n",
       "      <td>47</td>\n",
       "      <td>2017</td>\n",
       "      <td>2</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33464</th>\n",
       "      <td>5</td>\n",
       "      <td>75028</td>\n",
       "      <td>1391</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1590.628325</td>\n",
       "      <td>0</td>\n",
       "      <td>2022</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>27696 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       category  likeCount  commentCount  duration  made_for_kids   \n",
       "30735         3       1979           817         2              0  \\\n",
       "9957          5       7768           268         5              0   \n",
       "25156         1      36242          2948         2              0   \n",
       "17713        12       1430           381         1              0   \n",
       "16289        12      34432          1915         1              0   \n",
       "...         ...        ...           ...       ...            ...   \n",
       "2374          5      84648          8158         2              0   \n",
       "32682         3        489            39         1              0   \n",
       "6066          3     295356          8281         2              0   \n",
       "25250         1      35678          3664         1              0   \n",
       "33464         5      75028          1391         5              0   \n",
       "\n",
       "       viewPerComment  tagsCount  year  weekday  month  time_of_day   \n",
       "30735      151.157895         27  2020        1      8            1  \\\n",
       "9957      1254.167910         20  2017        5     12            1   \n",
       "25156      813.363976         38  2018        3      9            2   \n",
       "17713       50.144357          0  2013        2      9            0   \n",
       "16289      874.626110         13  2016        5     11            1   \n",
       "...               ...        ...   ...      ...    ...          ...   \n",
       "2374       400.011890         24  2016        1      5            1   \n",
       "32682      683.384615         26  2017        4      3            1   \n",
       "6066      1289.378819          6  2018        6     11            1   \n",
       "25250      802.075328         47  2017        2      6            2   \n",
       "33464     1590.628325          0  2022        5     12            1   \n",
       "\n",
       "       title_length  \n",
       "30735            96  \n",
       "9957             68  \n",
       "25156            62  \n",
       "17713            36  \n",
       "16289            45  \n",
       "...             ...  \n",
       "2374              5  \n",
       "32682            89  \n",
       "6066             39  \n",
       "25250            20  \n",
       "33464            31  \n",
       "\n",
       "[27696 rows x 12 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8cfef76",
   "metadata": {
    "id": "c8cfef76"
   },
   "source": [
    "## 1-c. Scaling data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "008a33bb",
   "metadata": {
    "id": "008a33bb"
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "scaler = MinMaxScaler()\n",
    "df_scaled = scaler.fit_transform(df)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "503c8ddf",
   "metadata": {
    "id": "503c8ddf"
   },
   "source": [
    "# 2. Model Implementation\n",
    "\n",
    "## 2-a. Random Forest"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7743d41c",
   "metadata": {
    "id": "7743d41c"
   },
   "source": [
    "**Initial Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "3db705f7",
   "metadata": {
    "id": "3db705f7"
   },
   "outputs": [],
   "source": [
    "# Initial Model \n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "RF = RandomForestClassifier(n_estimators=300, \n",
    "                             max_depth=3)\n",
    "\n",
    "# Fit RandomForestClassifier\n",
    "RF.fit(X_train, y_train)\n",
    "# Predict the test set labels\n",
    "y_pred = RF.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9a6a2778",
   "metadata": {
    "id": "9a6a2778",
    "outputId": "7324e5ea-c946-4da7-d9e0-753d468daa9d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test score:  0.6462093862815884\n",
      "Training score:  0.6547876949740035\n"
     ]
    }
   ],
   "source": [
    "print(\"Test score: \",RF.score(X_test, y_test))\n",
    "print(\"Training score: \", RF.score(X_train, y_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99c56f70",
   "metadata": {
    "id": "99c56f70"
   },
   "source": [
    "**Hyperparameter tuning with cross validation**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "90c84e1f",
   "metadata": {
    "id": "90c84e1f",
    "outputId": "de246f24-04bd-428d-978c-2ee45ab2e6af"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_split.py:676: UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Input \u001b[1;32mIn [10]\u001b[0m, in \u001b[0;36m<cell line: 12>\u001b[1;34m()\u001b[0m\n\u001b[0;32m     10\u001b[0m RF \u001b[38;5;241m=\u001b[39m RandomForestClassifier()\n\u001b[0;32m     11\u001b[0m grid_search \u001b[38;5;241m=\u001b[39m GridSearchCV(RF, param_grid, cv\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m5\u001b[39m)\n\u001b[1;32m---> 12\u001b[0m \u001b[43mgrid_search\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX_train\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my_train\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     14\u001b[0m best_RF \u001b[38;5;241m=\u001b[39m grid_search\u001b[38;5;241m.\u001b[39mbest_estimator_\n\u001b[0;32m     15\u001b[0m y_pred \u001b[38;5;241m=\u001b[39m best_RF\u001b[38;5;241m.\u001b[39mpredict(X_test)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:891\u001b[0m, in \u001b[0;36mBaseSearchCV.fit\u001b[1;34m(self, X, y, groups, **fit_params)\u001b[0m\n\u001b[0;32m    885\u001b[0m     results \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_format_results(\n\u001b[0;32m    886\u001b[0m         all_candidate_params, n_splits, all_out, all_more_results\n\u001b[0;32m    887\u001b[0m     )\n\u001b[0;32m    889\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m results\n\u001b[1;32m--> 891\u001b[0m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_run_search\u001b[49m\u001b[43m(\u001b[49m\u001b[43mevaluate_candidates\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    893\u001b[0m \u001b[38;5;66;03m# multimetric is determined here because in the case of a callable\u001b[39;00m\n\u001b[0;32m    894\u001b[0m \u001b[38;5;66;03m# self.scoring the return type is only known after calling\u001b[39;00m\n\u001b[0;32m    895\u001b[0m first_test_score \u001b[38;5;241m=\u001b[39m all_out[\u001b[38;5;241m0\u001b[39m][\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtest_scores\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:1392\u001b[0m, in \u001b[0;36mGridSearchCV._run_search\u001b[1;34m(self, evaluate_candidates)\u001b[0m\n\u001b[0;32m   1390\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_run_search\u001b[39m(\u001b[38;5;28mself\u001b[39m, evaluate_candidates):\n\u001b[0;32m   1391\u001b[0m     \u001b[38;5;124;03m\"\"\"Search all candidates in param_grid\"\"\"\u001b[39;00m\n\u001b[1;32m-> 1392\u001b[0m     \u001b[43mevaluate_candidates\u001b[49m\u001b[43m(\u001b[49m\u001b[43mParameterGrid\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mparam_grid\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:838\u001b[0m, in \u001b[0;36mBaseSearchCV.fit.<locals>.evaluate_candidates\u001b[1;34m(candidate_params, cv, more_results)\u001b[0m\n\u001b[0;32m    830\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mverbose \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[0;32m    831\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\n\u001b[0;32m    832\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFitting \u001b[39m\u001b[38;5;132;01m{0}\u001b[39;00m\u001b[38;5;124m folds for each of \u001b[39m\u001b[38;5;132;01m{1}\u001b[39;00m\u001b[38;5;124m candidates,\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    833\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m totalling \u001b[39m\u001b[38;5;132;01m{2}\u001b[39;00m\u001b[38;5;124m fits\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(\n\u001b[0;32m    834\u001b[0m             n_splits, n_candidates, n_candidates \u001b[38;5;241m*\u001b[39m n_splits\n\u001b[0;32m    835\u001b[0m         )\n\u001b[0;32m    836\u001b[0m     )\n\u001b[1;32m--> 838\u001b[0m out \u001b[38;5;241m=\u001b[39m \u001b[43mparallel\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    839\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdelayed\u001b[49m\u001b[43m(\u001b[49m\u001b[43m_fit_and_score\u001b[49m\u001b[43m)\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    840\u001b[0m \u001b[43m        \u001b[49m\u001b[43mclone\u001b[49m\u001b[43m(\u001b[49m\u001b[43mbase_estimator\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    841\u001b[0m \u001b[43m        \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    842\u001b[0m \u001b[43m        \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    843\u001b[0m \u001b[43m        \u001b[49m\u001b[43mtrain\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtrain\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    844\u001b[0m \u001b[43m        \u001b[49m\u001b[43mtest\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtest\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    845\u001b[0m \u001b[43m        \u001b[49m\u001b[43mparameters\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mparameters\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    846\u001b[0m \u001b[43m        \u001b[49m\u001b[43msplit_progress\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43msplit_idx\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mn_splits\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    847\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcandidate_progress\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mcand_idx\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mn_candidates\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    848\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mfit_and_score_kwargs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    849\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    850\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43;01mfor\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[43mcand_idx\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mparameters\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[43msplit_idx\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[43mtrain\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtest\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mproduct\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    851\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43menumerate\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mcandidate_params\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43menumerate\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mcv\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43msplit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mgroups\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    852\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    853\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    855\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(out) \u001b[38;5;241m<\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m    856\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    857\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNo fits were performed. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    858\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWas the CV iterator empty? \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    859\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWere there no candidates?\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    860\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:1046\u001b[0m, in \u001b[0;36mParallel.__call__\u001b[1;34m(self, iterable)\u001b[0m\n\u001b[0;32m   1043\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdispatch_one_batch(iterator):\n\u001b[0;32m   1044\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_iterating \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_original_iterator \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m-> 1046\u001b[0m \u001b[38;5;28;01mwhile\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdispatch_one_batch\u001b[49m\u001b[43m(\u001b[49m\u001b[43miterator\u001b[49m\u001b[43m)\u001b[49m:\n\u001b[0;32m   1047\u001b[0m     \u001b[38;5;28;01mpass\u001b[39;00m\n\u001b[0;32m   1049\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m pre_dispatch \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mall\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mor\u001b[39;00m n_jobs \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   1050\u001b[0m     \u001b[38;5;66;03m# The iterable was consumed all at once by the above for loop.\u001b[39;00m\n\u001b[0;32m   1051\u001b[0m     \u001b[38;5;66;03m# No need to wait for async callbacks to trigger to\u001b[39;00m\n\u001b[0;32m   1052\u001b[0m     \u001b[38;5;66;03m# consumption.\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:861\u001b[0m, in \u001b[0;36mParallel.dispatch_one_batch\u001b[1;34m(self, iterator)\u001b[0m\n\u001b[0;32m    859\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mFalse\u001b[39;00m\n\u001b[0;32m    860\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 861\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_dispatch\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtasks\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    862\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mTrue\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:779\u001b[0m, in \u001b[0;36mParallel._dispatch\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    777\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_lock:\n\u001b[0;32m    778\u001b[0m     job_idx \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlen\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jobs)\n\u001b[1;32m--> 779\u001b[0m     job \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_backend\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mapply_async\u001b[49m\u001b[43m(\u001b[49m\u001b[43mbatch\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcallback\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcb\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    780\u001b[0m     \u001b[38;5;66;03m# A job can complete so quickly than its callback is\u001b[39;00m\n\u001b[0;32m    781\u001b[0m     \u001b[38;5;66;03m# called before we get here, causing self._jobs to\u001b[39;00m\n\u001b[0;32m    782\u001b[0m     \u001b[38;5;66;03m# grow. To ensure correct results ordering, .insert is\u001b[39;00m\n\u001b[0;32m    783\u001b[0m     \u001b[38;5;66;03m# used (rather than .append) in the following line\u001b[39;00m\n\u001b[0;32m    784\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jobs\u001b[38;5;241m.\u001b[39minsert(job_idx, job)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py:208\u001b[0m, in \u001b[0;36mSequentialBackend.apply_async\u001b[1;34m(self, func, callback)\u001b[0m\n\u001b[0;32m    206\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mapply_async\u001b[39m(\u001b[38;5;28mself\u001b[39m, func, callback\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m):\n\u001b[0;32m    207\u001b[0m     \u001b[38;5;124;03m\"\"\"Schedule a func to be run\"\"\"\u001b[39;00m\n\u001b[1;32m--> 208\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[43mImmediateResult\u001b[49m\u001b[43m(\u001b[49m\u001b[43mfunc\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    209\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m callback:\n\u001b[0;32m    210\u001b[0m         callback(result)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py:572\u001b[0m, in \u001b[0;36mImmediateResult.__init__\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    569\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, batch):\n\u001b[0;32m    570\u001b[0m     \u001b[38;5;66;03m# Don't delay the application, to avoid keeping the input\u001b[39;00m\n\u001b[0;32m    571\u001b[0m     \u001b[38;5;66;03m# arguments in memory\u001b[39;00m\n\u001b[1;32m--> 572\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mresults \u001b[38;5;241m=\u001b[39m \u001b[43mbatch\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:262\u001b[0m, in \u001b[0;36mBatchedCalls.__call__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    258\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__call__\u001b[39m(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    259\u001b[0m     \u001b[38;5;66;03m# Set the default nested backend to self._backend but do not set the\u001b[39;00m\n\u001b[0;32m    260\u001b[0m     \u001b[38;5;66;03m# change the default number of processes to -1\u001b[39;00m\n\u001b[0;32m    261\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m parallel_backend(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backend, n_jobs\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_n_jobs):\n\u001b[1;32m--> 262\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m [func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    263\u001b[0m                 \u001b[38;5;28;01mfor\u001b[39;00m func, args, kwargs \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mitems]\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:262\u001b[0m, in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m    258\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__call__\u001b[39m(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    259\u001b[0m     \u001b[38;5;66;03m# Set the default nested backend to self._backend but do not set the\u001b[39;00m\n\u001b[0;32m    260\u001b[0m     \u001b[38;5;66;03m# change the default number of processes to -1\u001b[39;00m\n\u001b[0;32m    261\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m parallel_backend(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backend, n_jobs\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_n_jobs):\n\u001b[1;32m--> 262\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m [func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    263\u001b[0m                 \u001b[38;5;28;01mfor\u001b[39;00m func, args, kwargs \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mitems]\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\fixes.py:216\u001b[0m, in \u001b[0;36m_FuncWrapper.__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m    214\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__call__\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m    215\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m config_context(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig):\n\u001b[1;32m--> 216\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfunction(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:680\u001b[0m, in \u001b[0;36m_fit_and_score\u001b[1;34m(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, split_progress, candidate_progress, error_score)\u001b[0m\n\u001b[0;32m    678\u001b[0m         estimator\u001b[38;5;241m.\u001b[39mfit(X_train, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mfit_params)\n\u001b[0;32m    679\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 680\u001b[0m         estimator\u001b[38;5;241m.\u001b[39mfit(X_train, y_train, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mfit_params)\n\u001b[0;32m    682\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m:\n\u001b[0;32m    683\u001b[0m     \u001b[38;5;66;03m# Note fit time as time until error\u001b[39;00m\n\u001b[0;32m    684\u001b[0m     fit_time \u001b[38;5;241m=\u001b[39m time\u001b[38;5;241m.\u001b[39mtime() \u001b[38;5;241m-\u001b[39m start_time\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py:450\u001b[0m, in \u001b[0;36mBaseForest.fit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m    439\u001b[0m trees \u001b[38;5;241m=\u001b[39m [\n\u001b[0;32m    440\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_make_estimator(append\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m, random_state\u001b[38;5;241m=\u001b[39mrandom_state)\n\u001b[0;32m    441\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m i \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(n_more_estimators)\n\u001b[0;32m    442\u001b[0m ]\n\u001b[0;32m    444\u001b[0m \u001b[38;5;66;03m# Parallel loop: we prefer the threading backend as the Cython code\u001b[39;00m\n\u001b[0;32m    445\u001b[0m \u001b[38;5;66;03m# for fitting the trees is internally releasing the Python GIL\u001b[39;00m\n\u001b[0;32m    446\u001b[0m \u001b[38;5;66;03m# making threading more efficient than multiprocessing in\u001b[39;00m\n\u001b[0;32m    447\u001b[0m \u001b[38;5;66;03m# that case. However, for joblib 0.12+ we respect any\u001b[39;00m\n\u001b[0;32m    448\u001b[0m \u001b[38;5;66;03m# parallel_backend contexts set at a higher level,\u001b[39;00m\n\u001b[0;32m    449\u001b[0m \u001b[38;5;66;03m# since correctness does not rely on using threads.\u001b[39;00m\n\u001b[1;32m--> 450\u001b[0m trees \u001b[38;5;241m=\u001b[39m \u001b[43mParallel\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    451\u001b[0m \u001b[43m    \u001b[49m\u001b[43mn_jobs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mn_jobs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    452\u001b[0m \u001b[43m    \u001b[49m\u001b[43mverbose\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mverbose\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    453\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43m_joblib_parallel_args\u001b[49m\u001b[43m(\u001b[49m\u001b[43mprefer\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mthreads\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    454\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    455\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdelayed\u001b[49m\u001b[43m(\u001b[49m\u001b[43m_parallel_build_trees\u001b[49m\u001b[43m)\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    456\u001b[0m \u001b[43m        \u001b[49m\u001b[43mt\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    457\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m    458\u001b[0m \u001b[43m        \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    459\u001b[0m \u001b[43m        \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    460\u001b[0m \u001b[43m        \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    461\u001b[0m \u001b[43m        \u001b[49m\u001b[43mi\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    462\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mlen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mtrees\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    463\u001b[0m \u001b[43m        \u001b[49m\u001b[43mverbose\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mverbose\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    464\u001b[0m \u001b[43m        \u001b[49m\u001b[43mclass_weight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mclass_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    465\u001b[0m \u001b[43m        \u001b[49m\u001b[43mn_samples_bootstrap\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mn_samples_bootstrap\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    466\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    467\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43;01mfor\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mi\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mt\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[38;5;28;43menumerate\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mtrees\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    468\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    470\u001b[0m \u001b[38;5;66;03m# Collect newly grown trees\u001b[39;00m\n\u001b[0;32m    471\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mestimators_\u001b[38;5;241m.\u001b[39mextend(trees)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:1046\u001b[0m, in \u001b[0;36mParallel.__call__\u001b[1;34m(self, iterable)\u001b[0m\n\u001b[0;32m   1043\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdispatch_one_batch(iterator):\n\u001b[0;32m   1044\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_iterating \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_original_iterator \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m-> 1046\u001b[0m \u001b[38;5;28;01mwhile\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdispatch_one_batch\u001b[49m\u001b[43m(\u001b[49m\u001b[43miterator\u001b[49m\u001b[43m)\u001b[49m:\n\u001b[0;32m   1047\u001b[0m     \u001b[38;5;28;01mpass\u001b[39;00m\n\u001b[0;32m   1049\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m pre_dispatch \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mall\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mor\u001b[39;00m n_jobs \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m   1050\u001b[0m     \u001b[38;5;66;03m# The iterable was consumed all at once by the above for loop.\u001b[39;00m\n\u001b[0;32m   1051\u001b[0m     \u001b[38;5;66;03m# No need to wait for async callbacks to trigger to\u001b[39;00m\n\u001b[0;32m   1052\u001b[0m     \u001b[38;5;66;03m# consumption.\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:861\u001b[0m, in \u001b[0;36mParallel.dispatch_one_batch\u001b[1;34m(self, iterator)\u001b[0m\n\u001b[0;32m    859\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mFalse\u001b[39;00m\n\u001b[0;32m    860\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 861\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_dispatch\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtasks\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    862\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;01mTrue\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:779\u001b[0m, in \u001b[0;36mParallel._dispatch\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    777\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_lock:\n\u001b[0;32m    778\u001b[0m     job_idx \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlen\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jobs)\n\u001b[1;32m--> 779\u001b[0m     job \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_backend\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mapply_async\u001b[49m\u001b[43m(\u001b[49m\u001b[43mbatch\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcallback\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcb\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    780\u001b[0m     \u001b[38;5;66;03m# A job can complete so quickly than its callback is\u001b[39;00m\n\u001b[0;32m    781\u001b[0m     \u001b[38;5;66;03m# called before we get here, causing self._jobs to\u001b[39;00m\n\u001b[0;32m    782\u001b[0m     \u001b[38;5;66;03m# grow. To ensure correct results ordering, .insert is\u001b[39;00m\n\u001b[0;32m    783\u001b[0m     \u001b[38;5;66;03m# used (rather than .append) in the following line\u001b[39;00m\n\u001b[0;32m    784\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jobs\u001b[38;5;241m.\u001b[39minsert(job_idx, job)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py:208\u001b[0m, in \u001b[0;36mSequentialBackend.apply_async\u001b[1;34m(self, func, callback)\u001b[0m\n\u001b[0;32m    206\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mapply_async\u001b[39m(\u001b[38;5;28mself\u001b[39m, func, callback\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m):\n\u001b[0;32m    207\u001b[0m     \u001b[38;5;124;03m\"\"\"Schedule a func to be run\"\"\"\u001b[39;00m\n\u001b[1;32m--> 208\u001b[0m     result \u001b[38;5;241m=\u001b[39m \u001b[43mImmediateResult\u001b[49m\u001b[43m(\u001b[49m\u001b[43mfunc\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    209\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m callback:\n\u001b[0;32m    210\u001b[0m         callback(result)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py:572\u001b[0m, in \u001b[0;36mImmediateResult.__init__\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    569\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, batch):\n\u001b[0;32m    570\u001b[0m     \u001b[38;5;66;03m# Don't delay the application, to avoid keeping the input\u001b[39;00m\n\u001b[0;32m    571\u001b[0m     \u001b[38;5;66;03m# arguments in memory\u001b[39;00m\n\u001b[1;32m--> 572\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mresults \u001b[38;5;241m=\u001b[39m \u001b[43mbatch\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:262\u001b[0m, in \u001b[0;36mBatchedCalls.__call__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    258\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__call__\u001b[39m(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    259\u001b[0m     \u001b[38;5;66;03m# Set the default nested backend to self._backend but do not set the\u001b[39;00m\n\u001b[0;32m    260\u001b[0m     \u001b[38;5;66;03m# change the default number of processes to -1\u001b[39;00m\n\u001b[0;32m    261\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m parallel_backend(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backend, n_jobs\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_n_jobs):\n\u001b[1;32m--> 262\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m [func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    263\u001b[0m                 \u001b[38;5;28;01mfor\u001b[39;00m func, args, kwargs \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mitems]\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\joblib\\parallel.py:262\u001b[0m, in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m    258\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__call__\u001b[39m(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    259\u001b[0m     \u001b[38;5;66;03m# Set the default nested backend to self._backend but do not set the\u001b[39;00m\n\u001b[0;32m    260\u001b[0m     \u001b[38;5;66;03m# change the default number of processes to -1\u001b[39;00m\n\u001b[0;32m    261\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m parallel_backend(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_backend, n_jobs\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_n_jobs):\n\u001b[1;32m--> 262\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m [func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    263\u001b[0m                 \u001b[38;5;28;01mfor\u001b[39;00m func, args, kwargs \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mitems]\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\fixes.py:216\u001b[0m, in \u001b[0;36m_FuncWrapper.__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m    214\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__call__\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m    215\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m config_context(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mconfig):\n\u001b[1;32m--> 216\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfunction(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py:185\u001b[0m, in \u001b[0;36m_parallel_build_trees\u001b[1;34m(tree, forest, X, y, sample_weight, tree_idx, n_trees, verbose, class_weight, n_samples_bootstrap)\u001b[0m\n\u001b[0;32m    182\u001b[0m     \u001b[38;5;28;01melif\u001b[39;00m class_weight \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbalanced_subsample\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m    183\u001b[0m         curr_sample_weight \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m=\u001b[39m compute_sample_weight(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbalanced\u001b[39m\u001b[38;5;124m\"\u001b[39m, y, indices\u001b[38;5;241m=\u001b[39mindices)\n\u001b[1;32m--> 185\u001b[0m     \u001b[43mtree\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcurr_sample_weight\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcheck_input\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m)\u001b[49m\n\u001b[0;32m    186\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    187\u001b[0m     tree\u001b[38;5;241m.\u001b[39mfit(X, y, sample_weight\u001b[38;5;241m=\u001b[39msample_weight, check_input\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:937\u001b[0m, in \u001b[0;36mDecisionTreeClassifier.fit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m    899\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mfit\u001b[39m(\n\u001b[0;32m    900\u001b[0m     \u001b[38;5;28mself\u001b[39m, X, y, sample_weight\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, check_input\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m, X_idx_sorted\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdeprecated\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    901\u001b[0m ):\n\u001b[0;32m    902\u001b[0m     \u001b[38;5;124;03m\"\"\"Build a decision tree classifier from the training set (X, y).\u001b[39;00m\n\u001b[0;32m    903\u001b[0m \n\u001b[0;32m    904\u001b[0m \u001b[38;5;124;03m    Parameters\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    934\u001b[0m \u001b[38;5;124;03m        Fitted estimator.\u001b[39;00m\n\u001b[0;32m    935\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 937\u001b[0m     \u001b[38;5;28;43msuper\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    938\u001b[0m \u001b[43m        \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    939\u001b[0m \u001b[43m        \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    940\u001b[0m \u001b[43m        \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    941\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcheck_input\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcheck_input\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    942\u001b[0m \u001b[43m        \u001b[49m\u001b[43mX_idx_sorted\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mX_idx_sorted\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    943\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    944\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:203\u001b[0m, in \u001b[0;36mBaseDecisionTree.fit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m    200\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_outputs_ \u001b[38;5;241m=\u001b[39m y\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m1\u001b[39m]\n\u001b[0;32m    202\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m is_classification:\n\u001b[1;32m--> 203\u001b[0m     \u001b[43mcheck_classification_targets\u001b[49m\u001b[43m(\u001b[49m\u001b[43my\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    204\u001b[0m     y \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mcopy(y)\n\u001b[0;32m    206\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclasses_ \u001b[38;5;241m=\u001b[39m []\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\multiclass.py:189\u001b[0m, in \u001b[0;36mcheck_classification_targets\u001b[1;34m(y)\u001b[0m\n\u001b[0;32m    178\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mcheck_classification_targets\u001b[39m(y):\n\u001b[0;32m    179\u001b[0m     \u001b[38;5;124;03m\"\"\"Ensure that target y is of a non-regression type.\u001b[39;00m\n\u001b[0;32m    180\u001b[0m \n\u001b[0;32m    181\u001b[0m \u001b[38;5;124;03m    Only the following target types (as defined in type_of_target) are allowed:\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    187\u001b[0m \u001b[38;5;124;03m    y : array-like\u001b[39;00m\n\u001b[0;32m    188\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 189\u001b[0m     y_type \u001b[38;5;241m=\u001b[39m \u001b[43mtype_of_target\u001b[49m\u001b[43m(\u001b[49m\u001b[43my\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    190\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m y_type \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m [\n\u001b[0;32m    191\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbinary\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    192\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmulticlass\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    195\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmultilabel-sequences\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    196\u001b[0m     ]:\n\u001b[0;32m    197\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mUnknown label type: \u001b[39m\u001b[38;5;132;01m%r\u001b[39;00m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m y_type)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\multiclass.py:322\u001b[0m, in \u001b[0;36mtype_of_target\u001b[1;34m(y)\u001b[0m\n\u001b[0;32m    319\u001b[0m     suffix \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m\"\u001b[39m  \u001b[38;5;66;03m# [1, 2, 3] or [[1], [2], [3]]\u001b[39;00m\n\u001b[0;32m    321\u001b[0m \u001b[38;5;66;03m# check float and contains non-integer float values\u001b[39;00m\n\u001b[1;32m--> 322\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m y\u001b[38;5;241m.\u001b[39mdtype\u001b[38;5;241m.\u001b[39mkind \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mand\u001b[39;00m np\u001b[38;5;241m.\u001b[39many(y \u001b[38;5;241m!=\u001b[39m \u001b[43my\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mastype\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mint\u001b[39;49m\u001b[43m)\u001b[49m):\n\u001b[0;32m    323\u001b[0m     \u001b[38;5;66;03m# [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]\u001b[39;00m\n\u001b[0;32m    324\u001b[0m     _assert_all_finite(y)\n\u001b[0;32m    325\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcontinuous\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m+\u001b[39m suffix\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Hyperameter grid\n",
    "param_grid = {\n",
    "    'n_estimators': [100, 200, 300, 400, 500],\n",
    "    'max_depth': [3, 6, 9, 12, None]\n",
    "}\n",
    "\n",
    "RF = RandomForestClassifier()\n",
    "grid_search = GridSearchCV(RF, param_grid, cv=5)\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "best_RF = grid_search.best_estimator_\n",
    "y_pred = best_RF.predict(X_test)\n",
    "\n",
    "# Evaluate\n",
    "test_score = best_RF.score(X_test, y_test)\n",
    "train_score = best_RF.score(X_train, y_train)\n",
    "\n",
    "print(\"Test score: \", test_score)\n",
    "print(\"Training score: \", train_score)\n",
    "print(\"Best parameters: \", grid_search.best_params_)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "707bed90",
   "metadata": {
    "id": "707bed90"
   },
   "source": [
    "**Random Forest Final Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f93f6ad4",
   "metadata": {
    "id": "f93f6ad4"
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "RF = RandomForestClassifier(n_estimators = 300, \n",
    "                             max_depth = None)\n",
    "\n",
    "RF.fit(X_train, y_train)\n",
    "y_pred = RF.predict(X_test)\n",
    "rf_accuracy_test = RF.score(X_test, y_test)\n",
    "rf_accuracy_train = RF.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25a3890c",
   "metadata": {
    "id": "25a3890c",
    "outputId": "8dafc56f-91f8-4d4e-ac3d-0d786de10062"
   },
   "outputs": [],
   "source": [
    "print(\"Random Forest Test score: \",rf_accuracy_test)\n",
    "print(\"Random Forest Training score: \", rf_accuracy_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a552658",
   "metadata": {
    "id": "3a552658",
    "outputId": "8063b98e-416b-4058-bd8c-b71295f2e4a4"
   },
   "outputs": [],
   "source": [
    "## Check precision, recall and F1 score\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score\n",
    "\n",
    "rf_precision = precision_score(y_test, y_pred, average='weighted')\n",
    "rf_recall = recall_score(y_test, y_pred, average='weighted')\n",
    "rf_f1 = f1_score(y_test, y_pred, average='weighted')\n",
    "\n",
    "\n",
    "print(\"Random Forest Precision:\", rf_precision)\n",
    "print(\"Random Forest Recall:\", rf_recall)\n",
    "print(\"Random Forest F1 score:\", rf_f1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa0923f5",
   "metadata": {
    "id": "aa0923f5"
   },
   "source": [
    "- The precision of the Random Forest model is about 0.9198, indicating that out of all the instances predicted as positive by the model, approximately 91.98% of them are actually true positives.\n",
    "- The recall of the Random Forest model is also around 0.9208, suggesting that the model is able to correctly identify approximately 92.08% of the positive instances in the test data\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3394e5c",
   "metadata": {
    "id": "a3394e5c"
   },
   "source": [
    "## 2-b. Gradient Boosting Classifier"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ebbe2ab",
   "metadata": {
    "id": "6ebbe2ab"
   },
   "source": [
    "**Initial Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8066ce60",
   "metadata": {
    "id": "8066ce60"
   },
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Input \u001b[1;32mIn [11]\u001b[0m, in \u001b[0;36m<cell line: 5>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mensemble\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m GradientBoostingClassifier\n\u001b[0;32m      3\u001b[0m GB \u001b[38;5;241m=\u001b[39m GradientBoostingClassifier(max_depth\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m5\u001b[39m,  n_estimators\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m100\u001b[39m)\n\u001b[1;32m----> 5\u001b[0m \u001b[43mGB\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX_train\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my_train\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m      6\u001b[0m prediction \u001b[38;5;241m=\u001b[39m GB\u001b[38;5;241m.\u001b[39mpredict(X_test)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_gb.py:586\u001b[0m, in \u001b[0;36mBaseGradientBoosting.fit\u001b[1;34m(self, X, y, sample_weight, monitor)\u001b[0m\n\u001b[0;32m    583\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_resize_state()\n\u001b[0;32m    585\u001b[0m \u001b[38;5;66;03m# fit the boosting stages\u001b[39;00m\n\u001b[1;32m--> 586\u001b[0m n_stages \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_fit_stages\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    587\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    588\u001b[0m \u001b[43m    \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    589\u001b[0m \u001b[43m    \u001b[49m\u001b[43mraw_predictions\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    590\u001b[0m \u001b[43m    \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    591\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_rng\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    592\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX_val\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    593\u001b[0m \u001b[43m    \u001b[49m\u001b[43my_val\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    594\u001b[0m \u001b[43m    \u001b[49m\u001b[43msample_weight_val\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    595\u001b[0m \u001b[43m    \u001b[49m\u001b[43mbegin_at_stage\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    596\u001b[0m \u001b[43m    \u001b[49m\u001b[43mmonitor\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    597\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    599\u001b[0m \u001b[38;5;66;03m# change shape of arrays after fit (early-stopping or additional ests)\u001b[39;00m\n\u001b[0;32m    600\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m n_stages \u001b[38;5;241m!=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mestimators_\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m0\u001b[39m]:\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_gb.py:663\u001b[0m, in \u001b[0;36mBaseGradientBoosting._fit_stages\u001b[1;34m(self, X, y, raw_predictions, sample_weight, random_state, X_val, y_val, sample_weight_val, begin_at_stage, monitor)\u001b[0m\n\u001b[0;32m    656\u001b[0m     old_oob_score \u001b[38;5;241m=\u001b[39m loss_(\n\u001b[0;32m    657\u001b[0m         y[\u001b[38;5;241m~\u001b[39msample_mask],\n\u001b[0;32m    658\u001b[0m         raw_predictions[\u001b[38;5;241m~\u001b[39msample_mask],\n\u001b[0;32m    659\u001b[0m         sample_weight[\u001b[38;5;241m~\u001b[39msample_mask],\n\u001b[0;32m    660\u001b[0m     )\n\u001b[0;32m    662\u001b[0m \u001b[38;5;66;03m# fit next stage of trees\u001b[39;00m\n\u001b[1;32m--> 663\u001b[0m raw_predictions \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_fit_stage\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    664\u001b[0m \u001b[43m    \u001b[49m\u001b[43mi\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    665\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    666\u001b[0m \u001b[43m    \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    667\u001b[0m \u001b[43m    \u001b[49m\u001b[43mraw_predictions\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    668\u001b[0m \u001b[43m    \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    669\u001b[0m \u001b[43m    \u001b[49m\u001b[43msample_mask\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    670\u001b[0m \u001b[43m    \u001b[49m\u001b[43mrandom_state\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    671\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX_csc\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    672\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX_csr\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    673\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    675\u001b[0m \u001b[38;5;66;03m# track deviance (= loss)\u001b[39;00m\n\u001b[0;32m    676\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m do_oob:\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_gb.py:246\u001b[0m, in \u001b[0;36mBaseGradientBoosting._fit_stage\u001b[1;34m(self, i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_csc, X_csr)\u001b[0m\n\u001b[0;32m    243\u001b[0m     sample_weight \u001b[38;5;241m=\u001b[39m sample_weight \u001b[38;5;241m*\u001b[39m sample_mask\u001b[38;5;241m.\u001b[39mastype(np\u001b[38;5;241m.\u001b[39mfloat64)\n\u001b[0;32m    245\u001b[0m X \u001b[38;5;241m=\u001b[39m X_csr \u001b[38;5;28;01mif\u001b[39;00m X_csr \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m X\n\u001b[1;32m--> 246\u001b[0m \u001b[43mtree\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mresidual\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcheck_input\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m)\u001b[49m\n\u001b[0;32m    248\u001b[0m \u001b[38;5;66;03m# update tree leaves\u001b[39;00m\n\u001b[0;32m    249\u001b[0m loss\u001b[38;5;241m.\u001b[39mupdate_terminal_regions(\n\u001b[0;32m    250\u001b[0m     tree\u001b[38;5;241m.\u001b[39mtree_,\n\u001b[0;32m    251\u001b[0m     X,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    258\u001b[0m     k\u001b[38;5;241m=\u001b[39mk,\n\u001b[0;32m    259\u001b[0m )\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:1315\u001b[0m, in \u001b[0;36mDecisionTreeRegressor.fit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m   1278\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mfit\u001b[39m(\n\u001b[0;32m   1279\u001b[0m     \u001b[38;5;28mself\u001b[39m, X, y, sample_weight\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, check_input\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m, X_idx_sorted\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdeprecated\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   1280\u001b[0m ):\n\u001b[0;32m   1281\u001b[0m     \u001b[38;5;124;03m\"\"\"Build a decision tree regressor from the training set (X, y).\u001b[39;00m\n\u001b[0;32m   1282\u001b[0m \n\u001b[0;32m   1283\u001b[0m \u001b[38;5;124;03m    Parameters\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1312\u001b[0m \u001b[38;5;124;03m        Fitted estimator.\u001b[39;00m\n\u001b[0;32m   1313\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m-> 1315\u001b[0m     \u001b[38;5;28;43msuper\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1316\u001b[0m \u001b[43m        \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1317\u001b[0m \u001b[43m        \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1318\u001b[0m \u001b[43m        \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msample_weight\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1319\u001b[0m \u001b[43m        \u001b[49m\u001b[43mcheck_input\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcheck_input\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1320\u001b[0m \u001b[43m        \u001b[49m\u001b[43mX_idx_sorted\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mX_idx_sorted\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1321\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1322\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:420\u001b[0m, in \u001b[0;36mBaseDecisionTree.fit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m    409\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    410\u001b[0m     builder \u001b[38;5;241m=\u001b[39m BestFirstTreeBuilder(\n\u001b[0;32m    411\u001b[0m         splitter,\n\u001b[0;32m    412\u001b[0m         min_samples_split,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    417\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmin_impurity_decrease,\n\u001b[0;32m    418\u001b[0m     )\n\u001b[1;32m--> 420\u001b[0m \u001b[43mbuilder\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbuild\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtree_\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43msample_weight\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    422\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_outputs_ \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m \u001b[38;5;129;01mand\u001b[39;00m is_classifier(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    423\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_classes_ \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_classes_[\u001b[38;5;241m0\u001b[39m]\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "\n",
    "GB = GradientBoostingClassifier(max_depth=5,  n_estimators=100)\n",
    "\n",
    "GB.fit(X_train, y_train)\n",
    "prediction = GB.predict(X_test)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb44e53f",
   "metadata": {
    "id": "fb44e53f",
    "outputId": "979b5dc2-188e-416c-d4b6-1672dec302fc"
   },
   "outputs": [],
   "source": [
    "print(\"Test score: \",GB.score(X_test, y_test))\n",
    "print(\"Training score: \", GB.score(X_train, y_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed17130e",
   "metadata": {
    "id": "ed17130e"
   },
   "source": [
    "**Hyperparameter tuning with cross validation**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58bdf612",
   "metadata": {
    "id": "58bdf612",
    "outputId": "20d0e52c-16b0-49d5-f0f9-ffc61980acd4"
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "# hyperparameter grid\n",
    "param_grid = {\n",
    "    'learning_rate': [0.1, 0.05, 0.01, 0.001],\n",
    "    'max_depth': [3, 5, 7, 9, 11],\n",
    "    'n_estimators': [50, 100, 150, 200, 250]\n",
    "}\n",
    "\n",
    "\n",
    "GB = GradientBoostingClassifier()\n",
    "grid_search = GridSearchCV(GB, param_grid, cv=5)\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "# print the cross-validation scores\n",
    "cv_results = grid_search.cv_results_\n",
    "for mean_score, params in zip(cv_results[\"mean_test_score\"], cv_results[\"params\"]):\n",
    "    print(\"Mean CV score:\", mean_score, \"Parameters:\", params)\n",
    "\n",
    "\n",
    "best_GB = grid_search.best_estimator_\n",
    "best_cv_score = grid_search.best_score_\n",
    "\n",
    "# predict the test \n",
    "prediction = best_GB.predict(X_test)\n",
    "\n",
    "\n",
    "# print(\"Test score:\", best_GB.score(X_test, y_test))\n",
    "# print(\"Training score:\", best_GB.score(X_train, y_train))\n",
    "# print(\"Best parameters:\", grid_search.best_params_)\n",
    "# print(\"Best CV score:\", best_score) # the mean cross-validation score of the best model.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1cff112",
   "metadata": {
    "id": "c1cff112",
    "outputId": "67eac2a4-8f0f-4090-ec04-e8b2cee9577d"
   },
   "outputs": [],
   "source": [
    "print(\"Test score:\", best_GB.score(X_test, y_test))\n",
    "print(\"Training score:\", best_GB.score(X_train, y_train))\n",
    "print(\"Best parameters:\", grid_search.best_params_)\n",
    "print(\"Best CV score:\", best_cv_score) # the mean cross-validation score of the best model.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a17085f",
   "metadata": {
    "id": "9a17085f"
   },
   "source": [
    "**Gradient Boost Final Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2b1a41b",
   "metadata": {
    "id": "e2b1a41b",
    "outputId": "4ddfff29-c0b8-4e0b-95e8-db564d9889f5"
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "GB = GradientBoostingClassifier(learning_rate=0.1, max_depth=9, n_estimators=250)\n",
    "\n",
    "GB.fit(X_train, y_train)\n",
    "prediction = GB.predict(X_test)\n",
    "gb_accuracy_test = GB.score(X_test, y_test)\n",
    "gb_accuracy_train = GB.score(X_train, y_train)\n",
    "\n",
    "print(\"Test score: \", gb_accuracy_test)\n",
    "print(\"Training score: \", gb_accuracy_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce9fe755",
   "metadata": {
    "id": "ce9fe755",
    "outputId": "67a7c106-7bfd-4581-d672-8d47baf17e7b"
   },
   "outputs": [],
   "source": [
    "## Check precision, recall, and F1 score\n",
    "gb_precision = precision_score(y_test, prediction, average='weighted')\n",
    "gb_recall = recall_score(y_test, prediction, average='weighted')\n",
    "gb_f1 = f1_score(y_test, prediction, average='weighted')\n",
    "\n",
    "print(\"Gradient Boosting Classifier Precision:\", gb_precision)\n",
    "print(\"Gradient Boosting Classifier Recall:\", gb_recall)\n",
    "print(\"Gradient Boosting Classifier F1 score:\", gb_f1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4766677",
   "metadata": {
    "id": "e4766677"
   },
   "source": [
    "- The precision score of 0.9637 indicates that out of all instances predicted as positive by the model, approximately 96.37% of them are actually true positives. This indicates a high level of accuracy in predicting positive instances.\n",
    "\n",
    "- The recall score of 0.9639 also suggests that the model correctly identifies approximately 96.39% of the positive instances present in the test data. This indicates a high level of sensitivity in detecting positive instances."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76880b25",
   "metadata": {
    "id": "76880b25"
   },
   "source": [
    "## 2-c. Nueral Network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3184df6",
   "metadata": {
    "id": "e3184df6"
   },
   "outputs": [],
   "source": [
    "from sklearn.neural_network import MLPClassifier\n",
    "\n",
    "MLP = MLPClassifier(solver='adam', activation='logistic', alpha=0.4, tol=1e-5,\n",
    "                        hidden_layer_sizes=(100, 20), max_iter=500)\n",
    "MLP.fit(X_train, y_train)\n",
    "prediction = MLP.predict(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67977ead",
   "metadata": {
    "id": "67977ead",
    "outputId": "6cdf62a5-6bad-4a82-d8d0-3397d98a8c6d"
   },
   "outputs": [],
   "source": [
    "print(\"Test score: \",MLP.score(X_test, y_test))\n",
    "print(\"Training score: \", MLP.score(X_train, y_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9005f7ce",
   "metadata": {
    "id": "9005f7ce"
   },
   "source": [
    "**Hyperparameter tuning with cross validation**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e90c4d28",
   "metadata": {
    "id": "e90c4d28",
    "outputId": "2ac44ad8-5bff-4560-9591-60e1981504be"
   },
   "outputs": [],
   "source": [
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "# Define the parameter grid for hyperparameter tuning\n",
    "param_grid = {\n",
    "    'solver': ['adam'],\n",
    "    'activation': ['logistic', 'relu'],\n",
    "    'alpha': [0.1, 0.2, 0.3],\n",
    "    'hidden_layer_sizes': [(100,), (100, 50), (50, 20)],\n",
    "    'max_iter': [200, 300, 400]\n",
    "}\n",
    "\n",
    "\n",
    "mlp = MLPClassifier()\n",
    "grid_search = GridSearchCV(mlp, param_grid, cv=5)\n",
    "\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "best_model = grid_search.best_estimator_\n",
    "best_params = grid_search.best_params_\n",
    "\n",
    "\n",
    "test_score = best_model.score(X_test, y_test)\n",
    "train_score = best_model.score(X_train, y_train)\n",
    "\n",
    "print(\"Test score:\", test_score)\n",
    "print(\"Training score:\", train_score)\n",
    "print(\"Best parameters:\", best_params)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "45dbf66c",
   "metadata": {
    "id": "45dbf66c"
   },
   "source": [
    "**Neural Network Final Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3d69c93",
   "metadata": {
    "id": "e3d69c93",
    "outputId": "8cc2546d-c7fa-46fb-8425-a3eed7f01013"
   },
   "outputs": [],
   "source": [
    "from sklearn.neural_network import MLPClassifier\n",
    "MLP = MLPClassifier(solver='adam', activation='relu', alpha=0.2,\n",
    "                        hidden_layer_sizes=(100, 50), max_iter=400)\n",
    "MLP.fit(X_train, y_train)\n",
    "prediction = MLP.predict(X_test)\n",
    "mlp_accuracy_test = GB.score(X_test, y_test)\n",
    "mlp_accuracy_train = GB.score(X_train, y_train)\n",
    "\n",
    "print(\"Test score: \", mlp_accuracy_test)\n",
    "print(\"Training score: \", mlp_accuracy_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e73ea0a",
   "metadata": {
    "id": "8e73ea0a",
    "outputId": "f6912a82-e133-4125-a703-80ee4bb29d00"
   },
   "outputs": [],
   "source": [
    "## Check precision and recall \n",
    "mlp_precision = precision_score(y_test, prediction, average='weighted')\n",
    "mlp_recall = recall_score(y_test, prediction, average='weighted')\n",
    "mlp_f1 = f1_score(y_test, prediction, average='weighted')\n",
    "\n",
    "print(\"Nerual Network Precision:\", mlp_precision)\n",
    "print(\"Nerual Network Recall:\", mlp_recall)\n",
    "print(\"Nerual Network F1 score:\", mlp_f1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb8e76ea",
   "metadata": {
    "id": "cb8e76ea"
   },
   "source": [
    "- The precision score of 0.7391 suggests that out of all instances predicted as positive by the model, approximately 73.91% of them are actually true positives. This indicates the model's ability to correctly identify positive instances while minimizing false positives.\n",
    "\n",
    "- The recall score of 0.8296 suggests that the model correctly identifies approximately 82.96% of the positive instances present in the test data. This indicates the model's ability to capture a large portion of the positive instances and avoid false negatives.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef8d2947",
   "metadata": {
    "id": "ef8d2947"
   },
   "source": [
    "## 2-d. XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "be960de4",
   "metadata": {
    "id": "be960de4"
   },
   "outputs": [],
   "source": [
    "import xgboost as xgb\n",
    "\n",
    "first_XBG = xgb.XGBClassifier()\n",
    "\n",
    "first_XBG.fit(X_train, y_train)\n",
    "first_prediction = first_XBG.predict(X_test)\n",
    "\n",
    "xgb_accuracy_test = first_XBG.score(X_test, y_test)\n",
    "xgb_accuracy_train = first_XBG.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "764ec5df",
   "metadata": {
    "id": "764ec5df",
    "outputId": "56aaa821-5a41-4885-bef0-5e5deddc1f12"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test score:  0.9667870036101083\n",
      "Training score:  1.0\n"
     ]
    }
   ],
   "source": [
    "print(\"Test score: \", xgb_accuracy_test)\n",
    "print(\"Training score: \", xgb_accuracy_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77a8fc44",
   "metadata": {
    "id": "77a8fc44"
   },
   "source": [
    "**Hyperparameter tuning with cross validation**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53128fda",
   "metadata": {
    "id": "53128fda",
    "outputId": "2d3dcab8-e373-49d5-963d-b104a40b3e4f"
   },
   "outputs": [],
   "source": [
    "import xgboost as xgb\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "param_grid = {\n",
    "    'max_depth': [3, 5, 7],\n",
    "    'n_estimators': [100, 200, 300],\n",
    "    'learning_rate': [0.1, 0.01, 0.001]\n",
    "}\n",
    "\n",
    "\n",
    "XBG = xgb.XGBClassifier()\n",
    "\n",
    "grid_search = GridSearchCV(XBG, param_grid, cv=5)\n",
    "\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "best_model = grid_search.best_estimator_\n",
    "best_params = grid_search.best_params_\n",
    "\n",
    "test_score = best_model.score(X_test, y_test)\n",
    "train_score = best_model.score(X_train, y_train)\n",
    "\n",
    "print(\"Test score:\", test_score)\n",
    "print(\"Training score:\", train_score)\n",
    "print(\"Best parameters:\", best_params)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f1fbd594",
   "metadata": {
    "id": "f1fbd594"
   },
   "source": [
    "Since the test score is very close to the test score from out initial model, we decided to choose the initial model for simplicity reason."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d5524d2",
   "metadata": {
    "id": "9d5524d2",
    "outputId": "d4e69e84-a86a-4f13-fc79-67263cc39625"
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_score, recall_score\n",
    "\n",
    "xgb_precision = precision_score(y_test, first_prediction, average='weighted')\n",
    "xgb_recall = recall_score(y_test, first_prediction, average='weighted')\n",
    "xgb_f1 = f1_score(y_test, first_prediction, average = 'weighted')\n",
    "\n",
    "print(\"XGBoost Classifier Precision:\", xgb_precision)\n",
    "print(\"XGBoost Classifier Recall:\", xgb_recall)\n",
    "print(\"XGBoost Classifier F1 score:\", xgb_f1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5d5c9eb",
   "metadata": {
    "id": "f5d5c9eb"
   },
   "source": [
    "# 3. Model Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1dfc690",
   "metadata": {
    "id": "d1dfc690",
    "outputId": "f78d48a2-0853-4a84-a743-c9f2b159f664"
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Define the models and their corresponding performance metrics\n",
    "models = ['RandomForest', 'GradientBoosting', 'MLP', 'XGBoost']\n",
    "accuracies = [rf_accuracy_test, gb_accuracy_test, mlp_accuracy_test, xgb_accuracy_test]\n",
    "precisions = [rf_precision, gb_precision, mlp_precision, xgb_precision]\n",
    "recalls = [rf_recall, gb_recall, mlp_recall, xgb_recall]\n",
    "f1_scores = [rf_f1, gb_f1, mlp_f1, xgb_f1]\n",
    "\n",
    "# Plot the metrics\n",
    "plt.figure(figsize=(10, 6))\n",
    "\n",
    "# Accuracy\n",
    "plt.subplot(221)\n",
    "plt.bar(models, accuracies)\n",
    "plt.title('Accuracy')\n",
    "plt.ylim(0, 1)\n",
    "\n",
    "# Precision\n",
    "plt.subplot(222)\n",
    "plt.bar(models, precisions)\n",
    "plt.title('Precision')\n",
    "plt.ylim(0, 1)\n",
    "\n",
    "# Recall\n",
    "plt.subplot(223)\n",
    "plt.bar(models, recalls)\n",
    "plt.title('Recall')\n",
    "plt.ylim(0, 1)\n",
    "\n",
    "# F1-score\n",
    "plt.subplot(224)\n",
    "plt.bar(models, f1_scores)\n",
    "plt.title('F1-score')\n",
    "plt.ylim(0, 1)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90d585cd",
   "metadata": {
    "id": "90d585cd"
   },
   "source": [
    "According to the above bar charts, gradient boosting classifier and XGBoost are the top 2 models with the best metrics. Although both Gradient Boosting and XGBoost show excellent performance, XGBoost is preferred since its initial model was picked. As a result, we chose the XGBoost model because to its performance, efficiency, and simplicity. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "584349c9",
   "metadata": {},
   "source": [
    "# 4. Saving our best model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b650e056",
   "metadata": {
    "id": "b650e056"
   },
   "outputs": [],
   "source": [
    "import pickle  #Initialize the flask App\n",
    "# Saving model to current directory\n",
    "# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.\n",
    "pickle.dump(first_XBG, open('model.pkl','wb'))\n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
