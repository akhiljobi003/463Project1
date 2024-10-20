import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple


def load_stock_data(symbol: str, period: str = '1y') -> List[Tuple[int, float]]:
    # Download data
    stock = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # For 1 year of data
    data = stock.history(start=start_date, end=end_date)

    # Format data as list of (timestamp, price) tuples
    formatted_data = [(int(date.timestamp()), float(row['Close']))
                      for date, row in data.iterrows()]

    return formatted_data


class FinancialAnalysisSystem:
    def __init__(self, data: List[Tuple[int, float]]):
        self.data = data  # List of tuples (timestamp, price)

    def merge_sort(self, arr: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        return self.merge(left, right)

    def merge(self, left: List[Tuple[int, float]], right: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i][0] < right[j][0]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def max_subarray(self, prices: List[float]) -> Tuple[int, int, float]:
        def max_crossing_subarray(prices: List[float], low: int, mid: int, high: int) -> Tuple[int, int, float]:
            left_sum = float('-inf')
            sum = 0
            max_left = mid
            for i in range(mid, low - 1, -1):
                sum += prices[i]
                if sum > left_sum:
                    left_sum = sum
                    max_left = i

            right_sum = float('-inf')
            sum = 0
            max_right = mid + 1
            for i in range(mid + 1, high + 1):
                sum += prices[i]
                if sum > right_sum:
                    right_sum = sum
                    max_right = i

            return (max_left, max_right, left_sum + right_sum)

        def max_subarray_recursive(prices: List[float], low: int, high: int) -> Tuple[int, int, float]:
            if low == high:
                return (low, high, prices[low])

            mid = (low + high) // 2
            left_low, left_high, left_sum = max_subarray_recursive(prices, low, mid)
            right_low, right_high, right_sum = max_subarray_recursive(prices, mid + 1, high)
            cross_low, cross_high, cross_sum = max_crossing_subarray(prices, low, mid, high)

            if left_sum >= right_sum and left_sum >= cross_sum:
                return (left_low, left_high, left_sum)
            elif right_sum >= left_sum and right_sum >= cross_sum:
                return (right_low, right_high, right_sum)
            else:
                return (cross_low, cross_high, cross_sum)

        return max_subarray_recursive(prices, 0, len(prices) - 1)

    def closest_pair(self, points: List[Tuple[int, float]]) -> Tuple[Tuple[int, float], Tuple[int, float], float]:
        def distance(p1: Tuple[int, float], p2: Tuple[int, float]) -> float:
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def closest_pair_recursive(points_x: List[Tuple[int, float]], points_y: List[Tuple[int, float]]) -> Tuple[
            Tuple[int, float], Tuple[int, float], float]:
            n = len(points_x)
            if n <= 3:
                return min(((points_x[i], points_x[j], distance(points_x[i], points_x[j]))
                            for i in range(n) for j in range(i + 1, n)),
                           key=lambda x: x[2])

            mid = n // 2
            midpoint = points_x[mid][0]
            points_y_left = [p for p in points_y if p[0] <= midpoint]
            points_y_right = [p for p in points_y if p[0] > midpoint]

            left = closest_pair_recursive(points_x[:mid], points_y_left)
            right = closest_pair_recursive(points_x[mid:], points_y_right)

            min_pair = left if left[2] < right[2] else right
            min_distance = min_pair[2]

            strip = [p for p in points_y if abs(p[0] - midpoint) < min_distance]
            for i in range(len(strip)):
                for j in range(i + 1, min(i + 7, len(strip))):
                    dist = distance(strip[i], strip[j])
                    if dist < min_distance:
                        min_pair = (strip[i], strip[j], dist)
                        min_distance = dist

            return min_pair

        points_x = sorted(points, key=lambda p: p[0])
        points_y = sorted(points, key=lambda p: p[1])
        return closest_pair_recursive(points_x, points_y)

    def analyze(self) -> dict:
        # Step 1: Sort the data
        sorted_data = self.merge_sort(self.data)
        prices = [price for _, price in sorted_data]

        # Step 2: Find period of maximum gain
        start, end, max_gain = self.max_subarray([prices[i + 1] - prices[i] for i in range(len(prices) - 1)])

        # Step 3: Detect anomalies
        anomaly_pair = self.closest_pair(self.data)

        # Step 4: Generate report
        report = {
            "max_gain_period": (sorted_data[start][0], sorted_data[end + 1][0]),
            "max_gain": max_gain,
            "anomaly": {
                "points": (anomaly_pair[0], anomaly_pair[1]),
                "distance": anomaly_pair[2]
            }
        }

        return report


# Example usage
if __name__ == "__main__":
    # Load real stock data
    apple_data = load_stock_data('AAPL')

    # Create an instance of FinancialAnalysisSystem
    fas = FinancialAnalysisSystem(apple_data)

    # Perform analysis
    analysis_report = fas.analyze()

    # Print results
    print("Financial Analysis Report:")
    print(
        f"Period of Maximum Gain: {datetime.fromtimestamp(analysis_report['max_gain_period'][0])} to {datetime.fromtimestamp(analysis_report['max_gain_period'][1])}")
    print(f"Maximum Gain: ${analysis_report['max_gain']:.2f}")
    print(f"Anomaly Detected between:")
    print(
        f"  Point 1: {datetime.fromtimestamp(analysis_report['anomaly']['points'][0][0])}, ${analysis_report['anomaly']['points'][0][1]:.2f}")
    print(
        f"  Point 2: {datetime.fromtimestamp(analysis_report['anomaly']['points'][1][0])}, ${analysis_report['anomaly']['points'][1][1]:.2f}")
    print(f"Anomaly Distance: {analysis_report['anomaly']['distance']:.2f}")
