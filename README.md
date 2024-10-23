# big-data-analytics_project
Data-preprocessing-using-Spark

The objective of this project is to preprocess the Global Terrorism Database  using Apache Spark for efficient large-scale data processing and analysis. The GTD dataset is extensive, containing detailed information about terrorist incidents worldwide. The preprocessing includes handling missing data, transforming columns, and generating insights such as the count of incidents by year, casualties, and distribution of attack types.

In my PySpark program for preprocessing and exploratory data analysis (EDA) of the Global Terrorism Database (GTD), I started by addressing missing values. For critical columns like casualties, nkill, and nwound, I used imputation techniques by replacing missing values with zeros to avoid data distortion. Additionally, I removed records with missing values in essential columns like country and attacktype to maintain the dataset's integrity.Next, I checked the dataset for duplicate records and ensured that only unique entries were retained for accurate analysis. Once the data was cleaned, I began exploring the distribution of various features. I grouped the data by year to analyze the number of incidents over time and created aggregations to examine the total and average casualties per year.For categorical analysis, I looked at the frequency of attack types using group-by queries, providing insights into the most common methods of terrorism. I also explored regional distributions by counting the number of incidents across different regions, which helped identify the most affected areas.
In terms of outlier detection, I calculated the maximum and minimum casualties for individual incidents, identifying the most severe attacks. Finally, I generated a correlation matrix (via summary statistics) to examine the relationships between key numerical predictors like casualties, nkill, and nwound, which allowed me to better understand patterns of attacks and their severity.


I will be usign to VMs in this project, so I configured my two VMs and installed and configured Apache spark.

First let me generate ssh key pair for connection between the two vms (master and worker) I generated ssh key pair in hadoop1 then copied the key to the other vm (hadoop 2)

Ssh-keygen -t rsa

cat.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

Then I configure Spark in the cluster mode.

I user tar spark package on hadoop1 and scp to hadoop2

On hadoop1

cd /opt

tar czf spark.tar.gz spark

scp spark.tar.gz root@hadoop2:/opt

On hadoop2

cd /opt

tar xvzf spark.tar.gz

To start spark

/opt/spark/sbin/start-all.sh

To submit a spark job I used Command

-- /opt/spark/bin/spark-submit --master spark://hadoop1:7077 /opt/preprocessing.py

Performance of the program

I observed that the duration for completing the action using only 1 Vm was approximately 1 minute. When I ran the command again to check for consistency, the time remained almost the same at 57 seconds. This indicates that running the Python code on a single VM (using Hadoop 1 only) consistently takes about 1 minute1.

The duration for completing a specific action using two VMs was about 27 seconds. This is approximately 2 times faster than the time taken when using only one VM, which was around 1 minute. Running the Python code using both VMs (Hadoop 1 and Hadoop 2) demonstrates a significant improvement in processing speed.
