from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from pyspark.ml.linalg import Vectors

def csv_to_rdd(data):
    """
    Convert a DataFrame to an RDD and normalized the data.

    Args:
        data (DataFrame): The input DataFrame.

    Returns:
        RDD: The converted and normalized RDD.
    """
    # taking only the points
    filtered_df = data.select(data.columns[:-1])
    data_list = filtered_df.collect()

    # Create an empty array with the same shape as the DataFrame
    data_array = np.empty((len(data_list), len(filtered_df.columns)), dtype=object)

    # Populate the array with the data from the DataFrame
    for i, row in enumerate(data_list):
        data_array[i] = row

    data = data_array
    scaler = MinMaxScaler()
    scaler.fit(data)
    numpy_array = scaler.transform(data)

    features_rdd = spark.sparkContext.parallelize(numpy_array)
    features_rdd = features_rdd.map(lambda row: [float(x) if i != 4 else int(x) for i, x in enumerate(row)]) # casting to float the features and the class to int
    return features_rdd
def assign_centroid(point,centroids):
    """
    Assigns a given point to its nearest centroid.

    Parameters:
    - point: A numpy array representing the coordinates of a point.
    - centroids: A list of numpy arrays representing the coordinates of centroids.

    Returns:
    - A tuple containing the coordinates of the assigned centroid as a tuple, and a tuple of the point and count (1).
    """
    temp_min = float('inf')
    center = None
    for centroid in centroids:
        distance = np.linalg.norm(point - centroid)
        if distance < temp_min:
            temp_min = distance
            center = centroid
    return (tuple(center),(np.array(point),1))

def assign_integer_labels(unique_points, all_points):
    """
    Assigns integer labels to all_points based on the order of unique_points.

    Parameters:
    - unique_points: A list of unique points.
    - all_points: A list of all points.

    Returns:
    - A list of integer labels corresponding to each point in all_points.
    """
    unique_dict = {point: label for label, point in enumerate(unique_points)}
    integer_labels = [unique_dict[point] for point in all_points]
    return integer_labels

def CT_calc(c_old,c_new):
    """
    Calculates the convergence threshold (CT) between two sets of centroids.

    Parameters:
    - c_old: A numpy array representing the old centroids.
    - c_new: A numpy array representing the new centroids.

    Returns:
    - The distance between the old and new centroids.
    """
    ct = 0
    dist = np.linalg.norm(c_old - c_new)
    return dist
    
def kmeans(dataset, K, CT=0.0001, I=30, Exp=10):
    """
    Performs the k-means clustering algorithm on a given dataset.

    Parameters:
    - dataset: A dataset in .csv format.
    - K: The number of clusters.
    - CT: Convergence threshold (default is set to 0.0001).
    - I: Number of iterations per experiment (default is set to 30).
    - Exp: Number of experiments (default is set to 10).

    Returns:
    - A string representation of the average Calinski-Harabasz (CH) score and adjusted Rand index (ARI) with their standard deviations.
    """
    original_class = dataset.rdd.map(lambda col : col[-1]).collect() # original given calsses
    features_data = csv_to_rdd(dataset) #normalize and extract only the features
    exp = []
    CH = []
    ARI = []
    for e in range(Exp):
        ct = CT
        num_features = len(features_data.takeSample(False,1)[0])  # how many coordinates
        centroids= np.array(features_data.takeSample(False,K)) # the initial centroids
        for i in range(I):
            if ct < CT: # stop condition in case of getting to the threshold
                break
            else:
                map_points = features_data.map(lambda point: assign_centroid(point,centroids)) # find the nearest centroid
                reduce_points = map_points.reduceByKey(lambda x,y: (tuple(x[0][i]+y[0][i] for i in range(len(x[0]))),x[1]+y[1])) # sum all the points in the same centroid
                res = reduce_points.mapValues(lambda x: tuple(x[0][i]/x[1] for i in range(len(x[0])))).sortByKey().collect() # calculate the new centroids
                new_centroids = [res[i][1] for i in range(len(res))] # extract omly the new centroids without the old ones
                ct = CT_calc(centroids,new_centroids) 
                centroids = np.array(new_centroids)
        predict = features_data.map(lambda point: assign_centroid(point,centroids)[0])
        distinct_labels = predict.distinct().collect()
        integer_labels = assign_integer_labels(distinct_labels, predict.collect()) # indexing the new predicting centroids
        CH.append(metrics.calinski_harabasz_score(np.array(features_data.collect()),integer_labels))
        ARI.append(metrics.adjusted_rand_score(original_class,integer_labels))
    
    CH_avg = np.average(CH) # CH avg
    CH_std = np.std(CH) # CH std
    ARI_avg = np.average(ARI) # ARI avg
    ARI_std = np.std(ARI) # ARI std
    return ("K = " + str(K) + " " + "CH:" + " " + "(" + str(CH_avg)+";"+str(CH_std)+")"+ " " + "ARI:" + " " + "(" + str(ARI_avg)+";"+str(ARI_std)+")")

# Load the iris dataset from a CSV file into a Spark DataFrame.
spark = SparkSession.builder.getOrCreate() # Create a Spark session
dataset_name = "iris"
irist_path = f'/FileStore/tables/{dataset_name}.csv'
iris = spark.read.csv(irist_path, header= True)

# Load the glass dataset from a CSV file into a Spark DataFrame.
dataset_name = "glass"
glass_path = f'/FileStore/tables/{dataset_name}.csv'
glass = spark.read.csv(glass_path, header= True)

# Load the parkinsons dataset from a CSV file into a Spark DataFrame.
dataset_name = "parkinsons"
parkinsons_path = f'/FileStore/tables/{dataset_name}.csv'
parkinsons = spark.read.csv(parkinsons_path, header= True)

# printing results
data_sets = [iris,glass,parkinsons]
data_names = ['iris', 'glass', 'parkinsons']
 
K = [2,3,4,5,7]
for data, data_name in zip(data_sets, data_names):
    print("-------------------" + "data : " + data_name + " -------------------")
    for k in K:
        print(kmeans(data,k))
