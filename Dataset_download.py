import os
import pandas as pd
import kagglehub


def download_dataset(target_folder="datasets"):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    dataset_path = os.path.join(target_folder, "ENB2012_data.xlsx")

    if os.path.exists(dataset_path):
        print("Датасет уже существует в папке", target_folder)
        return dataset_path

    try:
        print("Загрузка датасета ENB2012...")
        path = kagglehub.dataset_download("halilbrahimhatun/enb2012-data")

        for file in os.listdir(path):
            if file.endswith('.xlsx'):
                source_file = os.path.join(path, file)
                df = pd.read_excel(source_file)
                df.to_excel(dataset_path, index=False)
                print(f"Датасет сохранен в: {dataset_path}")
                return dataset_path

        raise FileNotFoundError("XLSX файл не найден в скачанном датасете")

    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return None


dataset_path = download_dataset("datasets")
if dataset_path:
    df = pd.read_excel(dataset_path)
    print("Датасет успешно загружен!")
    print(df.head())
