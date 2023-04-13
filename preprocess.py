import pandas as pd

vehicle_file_path = "dataset/vehicles.csv"


def price_filter(df):
    lower_bound, upper_bound = df['price'].quantile([0.10, 0.99])
    print(lower_bound, upper_bound)
    filtered_data = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    print("dropped count: ", df.shape[0] - filtered_data.shape[0])
    return filtered_data


def drop_year_lower_than(df, threshold=1990):
    filtered_data = df[df['year'] >= threshold]
    print("dropped count: ", df.shape[0] - filtered_data.shape[0])
    return filtered_data


def drop_missing_type(df):
    filtered_data = df[df['type'].notna()]
    filtered_data = filtered_data[filtered_data['type'].notnull()]
    print("dropped count: ", df.shape[0] - filtered_data.shape[0])
    return filtered_data


def drop_missing_fuel(df):
    filtered_data = df[df['fuel'].notna()]
    filtered_data = filtered_data[filtered_data['fuel'].notnull()]
    print("dropped count: ", df.shape[0] - filtered_data.shape[0])
    return filtered_data


def drop_missing_odometer(df):
    filtered_data = df[df['odometer'].notna()]
    filtered_data = filtered_data[filtered_data['odometer'].notnull()]
    print("dropped count: ", df.shape[0] - filtered_data.shape[0])
    return filtered_data


def drop_missing_manufacturer(df):
    filtered_data = df[df['odometer'].notna()]
    filtered_data = filtered_data[filtered_data['manufacturer'].notnull()]
    print("dropped count: ", df.shape[0] - filtered_data.shape[0])
    return filtered_data


def print_columns_has_missing_data(df):
    for column in df.columns:
        if df[column].isnull().any():
            print(column)
        if df[column].isna().any():
            print(column)


def drop_title_status(df):
    filtered_data = df[df['title_status'].notna()]
    filtered_data = filtered_data[filtered_data['title_status'].notnull()]
    print("dropped count: ", df.shape[0] - filtered_data.shape[0])
    return filtered_data


def fill_missing_transmission_with_automatic(df):
    df['transmission'] = df['transmission'].fillna('automatic')
    return df


def main():
    df = pd.read_csv(vehicle_file_path)
    non_relevant_columns = ["id", "url", "region_url", "image_url", "VIN", "posting_date"]
    out_of_scope_columns = ["image_url", "description", "model"]
    missing_data_columns = ["condition", "cylinders", "size", "county", "drive", "paint_color"]
    imply_columns = ["lat", "long", "region"]
    drop_columns = non_relevant_columns + out_of_scope_columns + missing_data_columns + imply_columns
    df = df.drop(columns=drop_columns)
    df = drop_year_lower_than(df)
    df = drop_missing_type(df)
    df = drop_missing_fuel(df)
    df = drop_missing_odometer(df)
    df = drop_missing_manufacturer(df)
    df = drop_title_status(df)
    df = fill_missing_transmission_with_automatic(df)
    print_columns_has_missing_data(df)
    df = price_filter(df)
    print(df.shape)
    # move price to the last column
    price = df.pop('price')
    df['price'] = price
    # drop duplicates
    df.drop_duplicates(inplace=True)
    print(df.shape)
    # shuffle data
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    # train test split
    train = df[:int(df.shape[0] * 0.8)]
    test = df[int(df.shape[0] * 0.8):]
    train.to_csv("dataset/train/train.csv", index=False)
    test.to_csv("dataset/test/test.csv", index=False)


if __name__ == "__main__":
    main()
