import os
import pandas as pd


if __name__ == '__main__':

    # find jp-10
    # df_path = os.path.join('result', 'find_jp10', 'find_C1CC2C3CCC(C3)C2C1.csv')
    # filtered_df_path = os.path.join('result', 'find_jp10', 'find_C1CC2C3CCC(C3)C2C1_in_paper.csv')
    # df = pd.read_csv(df_path)
    #
    # filtered_df = df[
    #     (0.95 < df['density/(g/cm3)']) & (df['density/(g/cm3)'] < 1.0) &
    #     (215 < df['Tm/K']) & (df['Tm/K'] < 240) &
    #     (42.0 < df['mass_calorific_value_h/(MJ/kg)']) & (df['mass_calorific_value_h/(MJ/kg)'] < 42.5) &
    #     (337.4 < df['ISP']) & (df['ISP'] < 337.8)
    # ]
    #
    # print(filtered_df)
    # filtered_df.to_csv(filtered_df_path)

    # design
    df_path = os.path.join('result', 'design', 'design.csv')
    filtered_df_path = os.path.join('result', 'design', 'design_in_paper.csv')
    df = pd.read_csv(df_path)

    filtered_df = df[
        (0.98 < df['density/(g/cm3)']) & (df['density/(g/cm3)'] < 1.1) &
        (200 < df['Tm/K']) & (df['Tm/K'] < 240) &
        (42.5 < df['mass_calorific_value_h/(MJ/kg)']) & (df['mass_calorific_value_h/(MJ/kg)'] < 43.5) &
        (337.5 < df['ISP']) & (df['ISP'] < 338.5)
        ]

    print(filtered_df)
    filtered_df.to_csv(filtered_df_path)