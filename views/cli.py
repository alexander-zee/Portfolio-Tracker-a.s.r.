import pandas as pd


class CLIView:
    """
    Command-line interface view for user interaction.
    """

    @staticmethod
    def show_menu() -> None:
        print("\n=== Portfolio Tracker Menu ===")
        print("1. Add asset")
        print("2. Remove asset")
        print("3. Show portfolio overview")
        print("4. Show sector allocation")
        print("5. Show asset class allocation")
        print("6. Plot historical prices")
        print("7. Plot normalized prices")
        print("8. Run Monte Carlo simulation")
        print("9. Exit")

    @staticmethod
    def prompt_string(message: str) -> str:
        while True:
            value = input(message).strip()
            if value:
                return value
            print("Input cannot be empty. Please try again.")

    @staticmethod
    def prompt_float(message: str) -> float:
        while True:
            value = input(message).strip()
            try:
                number = float(value)
                if number <= 0:
                    print("Value must be greater than 0. Please try again.")
                    continue
                return number
            except ValueError:
                print("Invalid number. Please enter a valid numeric value.")

    @staticmethod
    def show_message(message: str) -> None:
        print(f"\n{message}")

    @staticmethod
    def show_dataframe(df: pd.DataFrame, title: str | None = None) -> None:
        if title:
            print(f"\n=== {title} ===")

        if df.empty:
            print("No data available.")
            return

        display_df = df.copy()

        for column in display_df.columns:
            if pd.api.types.is_float_dtype(display_df[column]):
                if "weight" in column.lower() or "pct" in column.lower():
                    display_df[column] = display_df[column].map(
                        lambda x: f"{x:.2%}" if pd.notna(x) else "NaN"
                    )
                else:
                    display_df[column] = display_df[column].map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "NaN"
                    )

        print(display_df.to_string(index=False))