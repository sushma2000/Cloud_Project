# Use a pre-built PySpark Docker image
FROM jupyter/pyspark-notebook

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run inference.py when the container launches
CMD ["spark-submit", "--class", "com.example.SparkMLInference", "--master", "yarn", "spark-ml-1.0-SNAPSHOT.jar"]
