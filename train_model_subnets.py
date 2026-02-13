import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor


def read_and_expand_csv_2(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Skip header and process data
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                # Parse the line as a Python tuple
                parsed_row = ast.literal_eval(line)
                data.append(parsed_row)
            except (ValueError, SyntaxError) as e:
                print(f"Skipping malformed line: {line[:50]}... Error: {e}")

    # Initial DataFrame with complex columns
    # Structure: (architecture), [ricci_sum], [ricci_mean], [ricci_var], parameters
    df = pd.DataFrame(data, columns=[
        'architecture_tuple', 
        'ricci_sum_list', 
        'ricci_mean_list', 
        'ricci_var_list', 
        'parameters'
    ])
    
    # --- 1. Expand Architecture (36 columns) ---
    # Header format: 12*attention + 12*inter_hidden + 12*residual
    arch_cols = (
        [f'attention_{i+1}' for i in range(12)] + 
        [f'inter_hidden_{i+1}' for i in range(12)] + 
        [f'residual_{i+1}' for i in range(12)]
    )
    # Convert the tuple column into a new DataFrame
    arch_df = pd.DataFrame(df['architecture_tuple'].tolist(), columns=arch_cols)
    
    # --- 2. Expand Ricci Values (12 columns each) ---
    sum_cols = [f'ricci_sum_{i+1}' for i in range(12)]
    mean_cols = [f'ricci_mean_{i+1}' for i in range(12)]
    var_cols = [f'ricci_var_{i+1}' for i in range(12)]
    
    sum_df = pd.DataFrame(df['ricci_sum_list'].tolist(), columns=sum_cols)
    mean_df = pd.DataFrame(df['ricci_mean_list'].tolist(), columns=mean_cols)
    var_df = pd.DataFrame(df['ricci_var_list'].tolist(), columns=var_cols)
    
    # --- 3. Concatenate everything together ---
    final_df = pd.concat([
        arch_df,
        sum_df,
        mean_df,
        var_df,
        df[['parameters']]
    ], axis=1)
    
    return final_df

def read_and_expand_csv(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Skip header and process data
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                # Parse the line as a Python tuple
                parsed_row = ast.literal_eval(line)
                data.append(parsed_row)
            except (ValueError, SyntaxError) as e:
                print(f"Skipping malformed line: {line[:50]}... Error: {e}")

    # Initial DataFrame with complex columns
    # Structure: (architecture), metric, f1, [ricci_sum], [ricci_mean], [ricci_var], parameters
    df = pd.DataFrame(data, columns=[
        'architecture_tuple', 
        'metric', 
        'f1', 
        'ricci_sum_list', 
        'ricci_mean_list', 
        'ricci_var_list', 
        'parameters'
    ])
    
    # --- 1. Expand Architecture (36 columns) ---
    # Header format: 12*attention + 12*inter_hidden + 12*residual
    arch_cols = (
        [f'attention_{i+1}' for i in range(12)] + 
        [f'inter_hidden_{i+1}' for i in range(12)] + 
        [f'residual_{i+1}' for i in range(12)]
    )
    # Convert the tuple column into a new DataFrame
    arch_df = pd.DataFrame(df['architecture_tuple'].tolist(), columns=arch_cols)
    
    # --- 2. Expand Ricci Values (12 columns each) ---
    sum_cols = [f'ricci_sum_{i+1}' for i in range(12)]
    mean_cols = [f'ricci_mean_{i+1}' for i in range(12)]
    var_cols = [f'ricci_var_{i+1}' for i in range(12)]
    
    sum_df = pd.DataFrame(df['ricci_sum_list'].tolist(), columns=sum_cols)
    mean_df = pd.DataFrame(df['ricci_mean_list'].tolist(), columns=mean_cols)
    var_df = pd.DataFrame(df['ricci_var_list'].tolist(), columns=var_cols)
    
    # --- 3. Concatenate everything together ---
    final_df = pd.concat([
        arch_df,
        df[['metric', 'f1']],
        sum_df,
        mean_df,
        var_df,
        df[['parameters']]
    ], axis=1)
    
    return final_df

# Example Usage:
df = read_and_expand_csv('/lustre/hdd/LAS/jannesar-lab/msamani/OSF/random_subnet_metrics.csv')

# remove column metric
df = df.drop(columns=['metric'])
X = df.drop(columns=['f1'])
y = df['f1']

# # 3. Feature Selection / Importance
# # We train a preliminary model to see which columns matter most
# rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_selector.fit(X, y)

# # Get feature importances
# importances = pd.Series(rf_selector.feature_importances_, index=X.columns)
# top_features = importances.sort_values(ascending=False).head(40)

# print("Top 40 Important Features:")
# print(top_features)

# # (Optional) Visualize Feature Importance
# plt.figure(figsize=(10, 6))
# top_features.plot(kind='barh')
# plt.title("Top 40 Features for predicting F1")
# plt.xlabel("Importance Score")
# plt.show()
# plt.savefig('feature_importance.png')

# 4. Train Final Model
# Using the top 40 features for better generalization (optional step, or use all X)
# best_cols = importances.sort_values(ascending=False).head(40).index
# X_reduced = X[best_cols]



# list of attention columns
attention_cols = [f'attention_{i+1}' for i in range(12)]
# list of inter_hidden columns
inter_hidden_cols = [f'inter_hidden_{i+1}' for i in range(12)]
# list of residual columns
residual_cols = [f'residual_{i+1}' for i in range(12)]


# list of ricci_sum columns
ricci_sum_cols = [f'ricci_sum_{i+1}' for i in range(12)]
# list of ricci_mean columns
ricci_mean_cols = [f'ricci_mean_{i+1}' for i in range(12)]
# list of ricci_var columns
ricci_var_cols = [f'ricci_var_{i+1}' for i in range(12)]

# list of parameter column
parameter_col = ['parameters']

#X = df[inter_hidden_cols + [residual_cols[0]]]
X = df[ricci_sum_cols + [residual_cols[0]]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model = RandomForestRegressor(n_estimators=100, random_state=42)


# model = xgb.XGBRegressor(
#     n_estimators=200, 
#     learning_rate=0.05, 
#     max_depth=5, 
#     random_state=42
# )

model = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=4, 
    subsample=0.8,
    random_state=42
)


model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2_score(y_test, predictions)}")




all_subnets_df = pd.DataFrame()
prefix = '/lustre/hdd/LAS/jannesar-lab/msamani/OSF/'
files_name = ["random_s1.csv", "random_s2.csv", "random_s3.csv", "random_s4.csv"]
for file in files_name:
    df = read_and_expand_csv_2(prefix + file)
    all_subnets_df = pd.concat([all_subnets_df, df], ignore_index=True)

# remove duplicate rows based on the architecture columns
arch_cols = (
    [f'attention_{i+1}' for i in range(12)] + 
    [f'inter_hidden_{i+1}' for i in range(12)] + 
    [f'residual_{i+1}' for i in range(12)]
)
all_subnets_df = all_subnets_df.drop_duplicates(subset=arch_cols)

# predict f1 for all subnets using the trained model
X_all = all_subnets_df[ricci_sum_cols + [residual_cols[0]]]
all_subnets_df['predicted_f1'] = model.predict(X_all)

# save the dataframe with predicted f1 values to a new csv file
all_subnets_df.to_csv(prefix + 'all_subnets_with_predictions.csv', index=False)

