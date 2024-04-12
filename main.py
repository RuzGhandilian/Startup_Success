from Pipeline import Pipeline

if __name__ == "__main__":
    data_path = './data/data.csv'
    pipeline = Pipeline(data_path)
    pipeline.run_pipeline()
