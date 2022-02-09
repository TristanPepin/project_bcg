
import src

df = src.generate_df()
model = src.fit_churn_model(df,epochs=10)
src.print_info_model()
src.evaluate_model(df)