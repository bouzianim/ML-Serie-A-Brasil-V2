import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveScoreRangeAnalyzer:
    """
    🚀 COMPREHENSIVE SCORE RANGE ANALYSIS SYSTEM
    Features:
    - Analyzes accuracy across 3-20 closest scores
    - Both METHOD 1 (Most Probable) and METHOD 2 (Closest to Average)
    - H/D/A and O/U accuracy tracking per score range
    - Revolutionary pattern detection across all ranges
    - Optimal score count identification
    - Detailed performance breakdown
    """
    def __init__(self):
        self.results = []
        self.score_range_results = {}

    def poisson(self, k, lam):
        """Calculate Poisson probability"""
        try:
            return (math.exp(-lam) * lam**k) / math.factorial(k)
        except:
            return 0

    def odds_to_prob(self, o1, x, o2):
        """Convert 1X2 odds to probabilities"""
        try:
            p1 = 1 / o1
            px = 1 / x
            p2 = 1 / o2
            total = p1 + px + p2
            return p1 / total, px / total, p2 / total
        except:
            return 0.33, 0.33, 0.33

    def match_probs(self, lambda1, lambda2):
        """Calculate H/D/A probabilities from xG values"""
        max_goals = 10
        p_home, p_draw, p_away = 0, 0, 0
        try:
            for i in range(max_goals):
                for j in range(max_goals):
                    p = self.poisson(i, lambda1) * self.poisson(j, lambda2)
                    if i > j:
                        p_home += p
                    elif i == j:
                        p_draw += p
                    else:
                        p_away += p
        except:
            p_home, p_draw, p_away = 0.33, 0.33, 0.33
        return p_home, p_draw, p_away

    def estimate_xg_from_odds(self, odds_home, odds_draw, odds_away):
        """Convert 1X2 odds to xG values using optimization"""
        try:
            p_target_home, p_target_draw, p_target_away = self.odds_to_prob(odds_home, odds_draw, odds_away)
            min_error = float('inf')
            best_lambda1, best_lambda2 = 1.0, 1.0
            for l1 in [round(x * 0.1, 1) for x in range(5, 35)]:
                for l2 in [round(x * 0.1, 1) for x in range(5, 35)]:
                    ph, pd, pa = self.match_probs(l1, l2)
                    error = (ph - p_target_home)**2 + (pd - p_target_draw)**2 + (pa - p_target_away)**2
                    if error < min_error:
                        min_error = error
                        best_lambda1 = l1
                        best_lambda2 = l2
            return round(best_lambda1, 2), round(best_lambda2, 2), min_error
        except:
            return 1.5, 1.5, 0.1

    def calculate_score_probabilities(self, lambda_home, lambda_away):
        """Calculate all score probabilities using Poisson distribution"""
        try:
            prob_home = {}
            prob_away = {}
            for i in range(10):
                prob_home[i] = self.poisson(i, lambda_home)
                prob_away[i] = self.poisson(i, lambda_away)
            prob_result = {}
            for i in range(10):
                prob_result[i] = {}
                for j in range(10):
                    prob_result[i][j] = prob_home[i] * prob_away[j]
            average_prob_result = sum(prob_result[i][j] for i in range(10) for j in range(10)) / 100
            max_prob_home = max(prob_home.values()) if prob_home.values() else 0.1
            max_prob_away = max(prob_away.values()) if prob_away.values() else 0.1
            sum_of_max_probs = (max_prob_home + max_prob_away) / 9
            return prob_home, prob_away, prob_result, average_prob_result, sum_of_max_probs
        except:
            # Return default values if calculation fails
            prob_result = {}
            for i in range(10):
                prob_result[i] = {}
                for j in range(10):
                    prob_result[i][j] = 0.01
            return {}, {}, prob_result, 0.01, 0.01

    def get_most_probable_scores(self, prob_result, num_scores):
        """METHOD 1: Get most probable scores by probability order"""
        try:
            all_scores = [(i, j, prob_result[i][j]) for i in range(10) for j in range(10)]
            most_probable = sorted(all_scores, key=lambda x: x[2], reverse=True)[:num_scores]
            return most_probable
        except:
            return [(1, 1, 0.1)] * min(num_scores, 5)

    def get_closest_to_average_scores(self, prob_result, sum_of_max_probs, num_scores):
        """METHOD 2: Get scores closest to average probability"""
        try:
            all_scores = [(i, j, prob_result[i][j]) for i in range(10) for j in range(10)]
            closest_to_avg = sorted(all_scores, key=lambda x: abs(x[2] - sum_of_max_probs))[:num_scores]
            return closest_to_avg
        except:
            return [(1, 1, 0.1)] * min(num_scores, 5)

    def predict_hda_from_scores(self, scores):
        """Predict H/D/A from score list using probability weighting"""
        try:
            if not scores:
                return 'D', 33.33, {'H': 33.33, 'D': 33.33, 'A': 33.33}
            total_prob = sum(prob for _, _, prob in scores)
            h_prob = sum(prob for home, away, prob in scores if home > away)
            d_prob = sum(prob for home, away, prob in scores if home == away)
            a_prob = sum(prob for home, away, prob in scores if home < away)
            if total_prob > 0:
                h_prob /= total_prob
                d_prob /= total_prob
                a_prob /= total_prob
            if h_prob > d_prob and h_prob > a_prob:
                prediction = 'H'
                confidence = h_prob * 100
            elif d_prob > h_prob and d_prob > a_prob:
                prediction = 'D'
                confidence = d_prob * 100
            else:
                prediction = 'A'
                confidence = a_prob * 100
            return prediction, confidence, {'H': h_prob * 100, 'D': d_prob * 100, 'A': a_prob * 100}
        except:
            return 'D', 33.33, {'H': 33.33, 'D': 33.33, 'A': 33.33}

    def predict_ou_from_scores(self, scores):
        """Predict Over/Under 2.5 from score list"""
        try:
            if not scores:
                return 'UNDER 2.5', 50.0, {'OVER': 50.0, 'UNDER': 50.0}
            total_prob = sum(prob for _, _, prob in scores)
            over_prob = sum(prob for home, away, prob in scores if home + away > 2.5)
            under_prob = sum(prob for home, away, prob in scores if home + away <= 2.5)
            if total_prob > 0:
                over_prob /= total_prob
                under_prob /= total_prob
            if over_prob > under_prob:
                prediction = 'OVER 2.5'
                confidence = over_prob * 100
            else:
                prediction = 'UNDER 2.5'
                confidence = under_prob * 100
            return prediction, confidence, {'OVER': over_prob * 100, 'UNDER': under_prob * 100}
        except:
            return 'UNDER 2.5', 50.0, {'OVER': 50.0, 'UNDER': 50.0}

    def calculate_weighted_avg(self, scores):
        """Calculate weighted average for pattern analysis"""
        try:
            if not scores:
                return 0
            total_prob = sum(prob for _, _, prob in scores)
            if total_prob > 0:
                weighted_avg = sum((home + away) * prob for home, away, prob in scores) / total_prob
            else:
                weighted_avg = sum(home + away for home, away, _ in scores) / len(scores)
            return weighted_avg
        except:
            return 2.0

    def comprehensive_score_range_analysis(self, df, sample_size=None): # Allow None for sample_size
        """
        🚀 COMPREHENSIVE ANALYSIS ACROSS ALL SCORE RANGES (3-20)
        Modified to handle sample_size=None for full dataset analysis.
        """
        print("🚀 COMPREHENSIVE SCORE RANGE ANALYSIS")
        print("=" * 80)
        
        # Determine dataset to use
        total_matches = len(df)
        if sample_size is None or sample_size >= total_matches:
            df_sample = df.copy()
            print(f"📊 Analyzing ALL {len(df_sample)} matches")
        else:
            df_sample = df.sample(sample_size).copy()
            print(f"📊 Analyzing sample of {sample_size} matches from {total_matches} total")

        # Initialize results storage
        score_ranges = list(range(3, 21))  # 3 to 20 scores
        range_results = {}
        for num_scores in score_ranges:
            range_results[num_scores] = {
                'method1': {'hda_correct': 0, 'ou_correct': 0, 'total': 0, 'high_conf_hda': 0, 'high_conf_total': 0, 'high_avg_pattern': 0, 'high_avg_total': 0},
                'method2': {'hda_correct': 0, 'ou_correct': 0, 'total': 0}
            }
        processed_matches = 0
        print(f"\n🔄 Processing matches across {len(score_ranges)} score ranges...")
        for idx, (_, row) in enumerate(df_sample.iterrows()):
            # Skip if missing data
            if pd.isna(row.get('Odd Home')) or pd.isna(row.get('Odd Draw')) or pd.isna(row.get('Odd Away')):
                continue
            if pd.isna(row.get('HG')) or pd.isna(row.get('AG')):
                continue
            try:
                # Get actual results
                actual_home = int(row['HG'])
                actual_away = int(row['AG'])
                actual_hda = 'H' if actual_home > actual_away else 'A' if actual_home < actual_away else 'D'
                actual_ou = 'OVER 2.5' if actual_home + actual_away > 2.5 else 'UNDER 2.5'
                # Convert odds to xG
                lambda_home, lambda_away, _ = self.estimate_xg_from_odds(row['Odd Home'], row['Odd Draw'], row['Odd Away'])
                # Calculate score probabilities
                prob_home, prob_away, prob_result, avg_prob, sum_of_max_probs = self.calculate_score_probabilities(lambda_home, lambda_away)
                # Test each score range
                for num_scores in score_ranges:
                    # METHOD 1: Most probable scores
                    most_probable_scores = self.get_most_probable_scores(prob_result, num_scores)
                    hda_pred_m1, hda_conf_m1, _ = self.predict_hda_from_scores(most_probable_scores)
                    ou_pred_m1, ou_conf_m1, _ = self.predict_ou_from_scores(most_probable_scores)
                    weighted_avg_m1 = self.calculate_weighted_avg(most_probable_scores)
                    # METHOD 2: Closest to average scores
                    closest_avg_scores = self.get_closest_to_average_scores(prob_result, sum_of_max_probs, num_scores)
                    hda_pred_m2, hda_conf_m2, _ = self.predict_hda_from_scores(closest_avg_scores)
                    ou_pred_m2, ou_conf_m2, _ = self.predict_ou_from_scores(closest_avg_scores)
                    # Update METHOD 1 results
                    range_results[num_scores]['method1']['total'] += 1
                    if hda_pred_m1 == actual_hda:
                        range_results[num_scores]['method1']['hda_correct'] += 1
                    if ou_pred_m1 == actual_ou:
                        range_results[num_scores]['method1']['ou_correct'] += 1
                    # High confidence H/D/A tracking
                    if hda_conf_m1 >= 60:
                        range_results[num_scores]['method1']['high_conf_total'] += 1
                        if hda_pred_m1 == actual_hda:
                            range_results[num_scores]['method1']['high_conf_hda'] += 1
                    # High AVG pattern tracking
                    if weighted_avg_m1 >= 2.5:
                        range_results[num_scores]['method1']['high_avg_total'] += 1
                        if actual_ou == 'OVER 2.5':
                            range_results[num_scores]['method1']['high_avg_pattern'] += 1
                    # Update METHOD 2 results
                    range_results[num_scores]['method2']['total'] += 1
                    if hda_pred_m2 == actual_hda:
                        range_results[num_scores]['method2']['hda_correct'] += 1
                    if ou_pred_m2 == actual_ou:
                        range_results[num_scores]['method2']['ou_correct'] += 1
                processed_matches += 1
                if processed_matches % 50 == 0:
                    print(f"   ⚡ Processed {processed_matches} matches...")
            except Exception as e:
                # Optionally print error for debugging: print(f"Error processing match {idx}: {e}")
                continue
        print(f"\n✅ Analysis complete: {processed_matches} matches processed")
        # Calculate accuracy percentages
        final_results = {}
        for num_scores in score_ranges:
            m1_data = range_results[num_scores]['method1']
            m2_data = range_results[num_scores]['method2']
            if m1_data['total'] > 0:
                final_results[num_scores] = {
                    'method1': {
                        'hda_accuracy': (m1_data['hda_correct'] / m1_data['total']) * 100,
                        'ou_accuracy': (m1_data['ou_correct'] / m1_data['total']) * 100,
                        'high_conf_hda_accuracy': (m1_data['high_conf_hda'] / m1_data['high_conf_total']) * 100 if m1_data['high_conf_total'] > 0 else 0,
                        'high_avg_pattern_accuracy': (m1_data['high_avg_pattern'] / m1_data['high_avg_total']) * 100 if m1_data['high_avg_total'] > 0 else 0,
                        'high_conf_opportunities': m1_data['high_conf_total'],
                        'high_avg_opportunities': m1_data['high_avg_total'],
                        'total_matches': m1_data['total']
                    },
                    'method2': {
                        'hda_accuracy': (m2_data['hda_correct'] / m2_data['total']) * 100,
                        'ou_accuracy': (m2_data['ou_correct'] / m2_data['total']) * 100,
                        'total_matches': m2_data['total']
                    }
                }
        self.score_range_results = final_results
        return final_results

    def find_optimal_score_counts(self, results):
        """
        🎯 Find optimal score counts for different prediction types
        """
        print("\n🎯 FINDING OPTIMAL SCORE COUNTS")
        print("=" * 60)
        optimal_results = {
            'method1_hda': {'best_score_count': 0, 'best_accuracy': 0},
            'method1_ou': {'best_score_count': 0, 'best_accuracy': 0},
            'method2_hda': {'best_score_count': 0, 'best_accuracy': 0},
            'method2_ou': {'best_score_count': 0, 'best_accuracy': 0},
            'high_conf_hda': {'best_score_count': 0, 'best_accuracy': 0},
            'high_avg_pattern': {'best_score_count': 0, 'best_accuracy': 0}
        }
        for num_scores, data in results.items():
            # METHOD 1 H/D/A
            if data['method1']['hda_accuracy'] > optimal_results['method1_hda']['best_accuracy']:
                optimal_results['method1_hda']['best_accuracy'] = data['method1']['hda_accuracy']
                optimal_results['method1_hda']['best_score_count'] = num_scores
            # METHOD 1 O/U
            if data['method1']['ou_accuracy'] > optimal_results['method1_ou']['best_accuracy']:
                optimal_results['method1_ou']['best_accuracy'] = data['method1']['ou_accuracy']
                optimal_results['method1_ou']['best_score_count'] = num_scores
            # METHOD 2 H/D/A
            if data['method2']['hda_accuracy'] > optimal_results['method2_hda']['best_accuracy']:
                optimal_results['method2_hda']['best_accuracy'] = data['method2']['hda_accuracy']
                optimal_results['method2_hda']['best_score_count'] = num_scores
            # METHOD 2 O/U
            if data['method2']['ou_accuracy'] > optimal_results['method2_ou']['best_accuracy']:
                optimal_results['method2_ou']['best_accuracy'] = data['method2']['ou_accuracy']
                optimal_results['method2_ou']['best_score_count'] = num_scores
            # High Confidence H/D/A
            if data['method1']['high_conf_hda_accuracy'] > optimal_results['high_conf_hda']['best_accuracy']:
                optimal_results['high_conf_hda']['best_accuracy'] = data['method1']['high_conf_hda_accuracy']
                optimal_results['high_conf_hda']['best_score_count'] = num_scores
            # High AVG Pattern
            if data['method1']['high_avg_pattern_accuracy'] > optimal_results['high_avg_pattern']['best_accuracy']:
                optimal_results['high_avg_pattern']['best_accuracy'] = data['method1']['high_avg_pattern_accuracy']
                optimal_results['high_avg_pattern']['best_score_count'] = num_scores
        # Display optimal results
        print(f"🔥 METHOD 1 H/D/A: {optimal_results['method1_hda']['best_score_count']} scores → {optimal_results['method1_hda']['best_accuracy']:.1f}% accuracy")
        print(f"⚽ METHOD 1 O/U: {optimal_results['method1_ou']['best_score_count']} scores → {optimal_results['method1_ou']['best_accuracy']:.1f}% accuracy")
        print(f"🧠 METHOD 2 H/D/A: {optimal_results['method2_hda']['best_score_count']} scores → {optimal_results['method2_hda']['best_accuracy']:.1f}% accuracy")
        print(f"📊 METHOD 2 O/U: {optimal_results['method2_ou']['best_score_count']} scores → {optimal_results['method2_ou']['best_accuracy']:.1f}% accuracy")
        print(f"⚡ High Confidence H/D/A: {optimal_results['high_conf_hda']['best_score_count']} scores → {optimal_results['high_conf_hda']['best_accuracy']:.1f}% accuracy")
        print(f"🎯 High AVG Pattern: {optimal_results['high_avg_pattern']['best_score_count']} scores → {optimal_results['high_avg_pattern']['best_accuracy']:.1f}% accuracy")
        return optimal_results

    def analyze_score_range_stability(self, results):
        """
        📊 Analyze stability and variance across score ranges
        """
        print("\n📊 SCORE RANGE STABILITY ANALYSIS")
        print("=" * 60)
        # Extract accuracy data
        score_counts = list(results.keys())
        m1_hda_accuracies = [results[sc]['method1']['hda_accuracy'] for sc in score_counts]
        m1_ou_accuracies = [results[sc]['method1']['ou_accuracy'] for sc in score_counts]
        m2_hda_accuracies = [results[sc]['method2']['hda_accuracy'] for sc in score_counts]
        m2_ou_accuracies = [results[sc]['method2']['ou_accuracy'] for sc in score_counts]
        # Calculate stability metrics
        stability_analysis = {
            'method1_hda': {
                'variance': np.var(m1_hda_accuracies),
                'std_dev': np.std(m1_hda_accuracies),
                'range': max(m1_hda_accuracies) - min(m1_hda_accuracies),
                'most_stable_range': self.find_most_stable_range(score_counts, m1_hda_accuracies)
            },
            'method1_ou': {
                'variance': np.var(m1_ou_accuracies),
                'std_dev': np.std(m1_ou_accuracies),
                'range': max(m1_ou_accuracies) - min(m1_ou_accuracies),
                'most_stable_range': self.find_most_stable_range(score_counts, m1_ou_accuracies)
            },
            'method2_hda': {
                'variance': np.var(m2_hda_accuracies),
                'std_dev': np.std(m2_hda_accuracies),
                'range': max(m2_hda_accuracies) - min(m2_hda_accuracies),
                'most_stable_range': self.find_most_stable_range(score_counts, m2_hda_accuracies)
            },
            'method2_ou': {
                'variance': np.var(m2_ou_accuracies),
                'std_dev': np.std(m2_ou_accuracies),
                'range': max(m2_ou_accuracies) - min(m2_ou_accuracies),
                'most_stable_range': self.find_most_stable_range(score_counts, m2_ou_accuracies)
            }
        }
        print(f"📊 Stability Analysis (Lower variance = more stable):")
        print(f"   METHOD 1 H/D/A: Variance {stability_analysis['method1_hda']['variance']:.2f}, Range {stability_analysis['method1_hda']['range']:.1f}%")
        print(f"   METHOD 1 O/U: Variance {stability_analysis['method1_ou']['variance']:.2f}, Range {stability_analysis['method1_ou']['range']:.1f}%")
        print(f"   METHOD 2 H/D/A: Variance {stability_analysis['method2_hda']['variance']:.2f}, Range {stability_analysis['method2_hda']['range']:.1f}%")
        print(f"   METHOD 2 O/U: Variance {stability_analysis['method2_ou']['variance']:.2f}, Range {stability_analysis['method2_ou']['range']:.1f}%")
        return stability_analysis

    def find_most_stable_range(self, score_counts, accuracies, window_size=5):
        """Find most stable range using rolling window"""
        try:
            if len(accuracies) < window_size:
                return score_counts[len(accuracies)//2]
            min_variance = float('inf')
            most_stable_idx = 0
            for i in range(len(accuracies) - window_size + 1):
                window_variance = np.var(accuracies[i:i+window_size])
                if window_variance < min_variance:
                    min_variance = window_variance
                    most_stable_idx = i + window_size // 2
            return score_counts[most_stable_idx]
        except:
            return score_counts[len(score_counts)//2]

def create_score_range_accuracy_chart(results):
    """Create comprehensive score range accuracy chart"""
    try:
        score_counts = list(results.keys())
        # Extract accuracy data
        m1_hda = [results[sc]['method1']['hda_accuracy'] for sc in score_counts]
        m1_ou = [results[sc]['method1']['ou_accuracy'] for sc in score_counts]
        m2_hda = [results[sc]['method2']['hda_accuracy'] for sc in score_counts]
        m2_ou = [results[sc]['method2']['ou_accuracy'] for sc in score_counts]
        fig = go.Figure()
        # METHOD 1 lines
        fig.add_trace(go.Scatter(
            x=score_counts, y=m1_hda,
            mode='lines+markers',
            name='METHOD 1 H/D/A',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=score_counts, y=m1_ou,
            mode='lines+markers',
            name='METHOD 1 O/U',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=8, symbol='square')
        ))
        # METHOD 2 lines
        fig.add_trace(go.Scatter(
            x=score_counts, y=m2_hda,
            mode='lines+markers',
            name='METHOD 2 H/D/A',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=score_counts, y=m2_ou,
            mode='lines+markers',
            name='METHOD 2 O/U',
            line=dict(color='#4ecdc4', width=3, dash='dash'),
            marker=dict(size=8, symbol='square')
        ))
        # Add benchmark lines
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Random Baseline (50%)")
        fig.add_hline(y=60, line_dash="dot", line_color="orange", annotation_text="Good Performance (60%)")
        fig.add_hline(y=70, line_dash="dot", line_color="green", annotation_text="Excellent Performance (70%)")
        fig.update_layout(
            title='🚀 Comprehensive Score Range Accuracy Analysis (3-20 Scores)',
            xaxis_title='Number of Closest Scores Used',
            yaxis_title='Accuracy (%)',
            height=600,
            hovermode='x unified'
        )
        return fig
    except:
        fig = go.Figure()
        fig.add_annotation(text="Error creating chart", x=0.5, y=0.5)
        return fig

def create_revolutionary_patterns_by_range_chart(results):
    """Create revolutionary patterns performance by score range"""
    try:
        score_counts = list(results.keys())
        # Extract revolutionary pattern data
        high_conf_accuracies = []
        high_avg_accuracies = []
        high_conf_opportunities = []
        high_avg_opportunities = []
        for sc in score_counts:
            high_conf_accuracies.append(results[sc]['method1']['high_conf_hda_accuracy'])
            high_avg_accuracies.append(results[sc]['method1']['high_avg_pattern_accuracy'])
            high_conf_opportunities.append(results[sc]['method1']['high_conf_opportunities'])
            high_avg_opportunities.append(results[sc]['method1']['high_avg_opportunities'])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'High Confidence H/D/A Accuracy',
                'High AVG Pattern O/U Accuracy',
                'High Confidence Opportunities',
                'High AVG Pattern Opportunities'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # High Confidence H/D/A Accuracy
        fig.add_trace(go.Scatter(
            x=score_counts, y=high_conf_accuracies,
            mode='lines+markers',
            name='High Conf H/D/A',
            line=dict(color='#ff6b6b', width=4),
            marker=dict(size=10)
        ), row=1, col=1)

        # High AVG Pattern O/U Accuracy
        fig.add_trace(go.Scatter(
            x=score_counts, y=high_avg_accuracies,
            mode='lines+markers',
            name='High AVG Pattern',
            line=dict(color='#ffd93d', width=4),
            marker=dict(size=10)
        ), row=1, col=2)

        # Opportunities charts
        fig.add_trace(go.Bar(
            x=score_counts, y=high_conf_opportunities,
            name='High Conf Opps',
            marker_color='#ff6b6b'
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=score_counts, y=high_avg_opportunities,
            name='High AVG Opps',
            marker_color='#ffd93d'
        ), row=2, col=2)

        # Add benchmark lines for accuracy charts
        fig.add_hline(y=63.4, line_dash="dash", line_color="red",
                      annotation_text="Discovery Benchmark (63.4%)", row=1, col=1)
        fig.add_hline(y=75.0, line_dash="dash", line_color="gold",
                      annotation_text="Discovery Benchmark (75%)", row=1, col=2)

        fig.update_layout(
            title='⚡ Revolutionary Patterns Performance Across Score Ranges',
            height=600,
            showlegend=False
        )
        return fig

    except Exception as e: # Catch the specific error for better debugging if needed
        # print(f"Error in create_revolutionary_patterns_by_range_chart: {e}") # Optional debug print
        fig = go.Figure()
        fig.add_annotation(text="Error creating chart", x=0.5, y=0.5, showarrow=False)
        return fig

def create_optimal_score_count_summary(optimal_results):
    """Create optimal score count summary visualization"""
    try:
        categories = ['METHOD 1 H/D/A', 'METHOD 1 O/U', 'METHOD 2 H/D/A', 'METHOD 2 O/U',
                     'High Conf H/D/A', 'High AVG Pattern']
        score_counts = [
            optimal_results['method1_hda']['best_score_count'],
            optimal_results['method1_ou']['best_score_count'],
            optimal_results['method2_hda']['best_score_count'],
            optimal_results['method2_ou']['best_score_count'],
            optimal_results['high_conf_hda']['best_score_count'],
            optimal_results['high_avg_pattern']['best_score_count']
        ]
        accuracies = [
            optimal_results['method1_hda']['best_accuracy'],
            optimal_results['method1_ou']['best_accuracy'],
            optimal_results['method2_hda']['best_accuracy'],
            optimal_results['method2_ou']['best_accuracy'],
            optimal_results['high_conf_hda']['best_accuracy'],
            optimal_results['high_avg_pattern']['best_accuracy']
        ]
        colors = ['#ff6b6b', '#ff6b6b', '#4ecdc4', '#4ecdc4', '#ffd93d', '#a8e6cf']
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Optimal Score Counts', 'Maximum Accuracies Achieved'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        # Optimal score counts
        fig.add_trace(go.Bar(
            x=categories, y=score_counts,
            marker=dict(color=colors),
            text=[f"{sc} scores" for sc in score_counts],
            textposition='auto',
            name="Score Counts"
        ), row=1, col=1)
        # Maximum accuracies
        fig.add_trace(go.Bar(
            x=categories, y=accuracies,
            marker=dict(color=colors),
            text=[f"{acc:.1f}%" for acc in accuracies],
            textposition='auto',
            name="Accuracies"
        ), row=1, col=2)
        fig.update_layout(
            title='🎯 Optimal Score Count Analysis Summary',
            height=500,
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        return fig
    except:
        fig = go.Figure()
        fig.add_annotation(text="Error creating chart", x=0.5, y=0.5)
        return fig

def load_data():
    """Load the dataset with odds"""
    try:
        df = pd.read_csv('ml_dataset_CLEAN.csv')
        # Check for required columns
        required_cols = ['Odd Home', 'Odd Draw', 'Odd Away', 'HG', 'AG']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("Please ensure your CSV file has columns: 'Odd Home', 'Odd Draw', 'Odd Away', 'HG', 'AG'")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

def main():
    st.set_page_config(
        page_title="🚀 Comprehensive Score Range Analysis",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .optimal-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .revolutionary-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 3px solid #ff6b6b;
    }
    .stability-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    # Header
    st.markdown('<h1 class="main-header">🚀 Comprehensive Score Range Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666;">🎯 Analyze accuracy across 3-20 closest scores | 📊 Find optimal score counts | ⚡ Revolutionary pattern optimization</p>', unsafe_allow_html=True)
    # Load data
    df = load_data()
    if df is None:
        st.stop()

    total_matches = len(df)
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ComprehensiveScoreRangeAnalyzer()
    analyzer = st.session_state.analyzer
    # Sidebar
    st.sidebar.markdown("## 🎯 Analysis Configuration")
    
    # --- MODIFIED SLIDER ---
    # Use a special value (e.g., total_matches + 1000) to represent "All"
    # Or use None by checking if the selected value is greater than or equal to total_matches
    sample_size = st.sidebar.slider(
        "📊 Sample Size for Analysis",
        min_value=100,
        max_value=total_matches, # Allow up to the total number of matches
        value=min(300, total_matches), # Default to 300 or total if less
        step=50,
        help="Larger samples = more accurate but slower analysis. Select the maximum value to analyze ALL matches."
    )
    # Determine if we should analyze all matches
    # If the slider is at the maximum (total_matches), analyze all.
    use_all_data = (sample_size >= total_matches)
    effective_sample_size = None if use_all_data else sample_size
    # --- END MODIFIED SLIDER ---

    st.sidebar.markdown("### 🔍 What This Analysis Does")
    st.sidebar.info("""
    **Comprehensive Score Range Testing:**
    - Tests 3 to 20 closest scores
    - Both METHOD 1 & METHOD 2
    - H/D/A and O/U accuracy tracking
    - Revolutionary pattern optimization
    - Finds optimal score count per strategy
    - Stability analysis across ranges
    """)
    # Dataset info
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.info(f"""
    **Total Matches**: {total_matches:,}
    **Analysis Sample**: {"ALL" if use_all_data else sample_size} ({sample_size:,} matches)
    **Leagues**: {df['League'].nunique()}
    **Countries**: {df['Country'].nunique()}
    """)
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Run Analysis", "📊 Accuracy Charts", "🎯 Optimal Results", "📈 Stability Analysis"])
    with tab1:
        st.markdown("## 🚀 Comprehensive Score Range Analysis")
        st.markdown("### Analyze accuracy across different numbers of closest scores (3-20)")
        analysis_button_label = "🔄 Run Comprehensive Score Range Analysis on ALL Matches" if use_all_data else f"🔄 Run Analysis on {sample_size} Matches"
        if st.button(analysis_button_label, type="primary", use_container_width=True):
            with st.spinner(f"🔮 Running comprehensive analysis on {'ALL' if use_all_data else sample_size} matches across 18 score ranges..."):
                # --- PASS THE MODIFIED sample_size ARGUMENT ---
                # Run the comprehensive analysis. Pass None to analyze all.
                results = analyzer.comprehensive_score_range_analysis(df, effective_sample_size)
                # --- END MODIFIED ARGUMENT ---
                # Store results in session state
                st.session_state.score_range_results = results
                if results:
                    st.success(f"✅ Analysis complete! Analyzed {'ALL' if use_all_data else sample_size} matches across 18 score ranges (3-20)")
                    # Quick summary
                    st.markdown("### 📊 Quick Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        # Find best METHOD 1 H/D/A
                        best_m1_hda = max(results.items(), key=lambda x: x[1]['method1']['hda_accuracy'])
                        st.metric(
                            "🔥 Best METHOD 1 H/D/A",
                            f"{best_m1_hda[1]['method1']['hda_accuracy']:.1f}%",
                            delta=f"{best_m1_hda[0]} scores"
                        )
                    with col2:
                        # Find best METHOD 1 O/U
                        best_m1_ou = max(results.items(), key=lambda x: x[1]['method1']['ou_accuracy'])
                        st.metric(
                            "⚽ Best METHOD 1 O/U",
                            f"{best_m1_ou[1]['method1']['ou_accuracy']:.1f}%",
                            delta=f"{best_m1_ou[0]} scores"
                        )
                    with col3:
                        # Find best METHOD 2 H/D/A
                        best_m2_hda = max(results.items(), key=lambda x: x[1]['method2']['hda_accuracy'])
                        st.metric(
                            "🧠 Best METHOD 2 H/D/A",
                            f"{best_m2_hda[1]['method2']['hda_accuracy']:.1f}%",
                            delta=f"{best_m2_hda[0]} scores"
                        )
                    with col4:
                        # Find best METHOD 2 O/U
                        best_m2_ou = max(results.items(), key=lambda x: x[1]['method2']['ou_accuracy'])
                        st.metric(
                            "📊 Best METHOD 2 O/U",
                            f"{best_m2_ou[1]['method2']['ou_accuracy']:.1f}%",
                            delta=f"{best_m2_ou[0]} scores"
                        )
                    # Revolutionary patterns summary
                    st.markdown("### ⚡ Revolutionary Patterns Performance")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Find best high confidence H/D/A
                        best_high_conf = max(
                            [(k, v) for k, v in results.items() if v['method1']['high_conf_hda_accuracy'] > 0],
                            key=lambda x: x[1]['method1']['high_conf_hda_accuracy'],
                            default=(5, {'method1': {'high_conf_hda_accuracy': 0, 'high_conf_opportunities': 0}})
                        )
                        st.markdown(f"""
                        <div class="revolutionary-card">
                        <h3>🔥 HIGH CONFIDENCE H/D/A</h3>
                        <h2>{best_high_conf[1]['method1']['high_conf_hda_accuracy']:.1f}% accuracy</h2>
                        <p><strong>Optimal Score Count:</strong> {best_high_conf[0]} scores</p>
                        <p><strong>Opportunities:</strong> {best_high_conf[1]['method1']['high_conf_opportunities']} matches</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        # Find best high AVG pattern
                        best_high_avg = max(
                            [(k, v) for k, v in results.items() if v['method1']['high_avg_pattern_accuracy'] > 0],
                            key=lambda x: x[1]['method1']['high_avg_pattern_accuracy'],
                            default=(5, {'method1': {'high_avg_pattern_accuracy': 0, 'high_avg_opportunities': 0}})
                        )
                        st.markdown(f"""
                        <div class="revolutionary-card">
                        <h3>⚡ HIGH AVG PATTERN</h3>
                        <h2>{best_high_avg[1]['method1']['high_avg_pattern_accuracy']:.1f}% accuracy</h2>
                        <p><strong>Optimal Score Count:</strong> {best_high_avg[0]} scores</p>
                        <p><strong>Opportunities:</strong> {best_high_avg[1]['method1']['high_avg_opportunities']} matches</p>
                        </div>
                        """, unsafe_allow_html=True)
                    # Detailed results table
                    st.markdown("### 📋 Detailed Results Table")
                    # Create detailed results dataframe
                    detailed_data = []
                    for score_count, data in sorted(results.items()):
                        detailed_data.append({
                            'Score_Count': score_count,
                            'M1_HDA_Accuracy': f"{data['method1']['hda_accuracy']:.1f}%",
                            'M1_OU_Accuracy': f"{data['method1']['ou_accuracy']:.1f}%",
                            'M2_HDA_Accuracy': f"{data['method2']['hda_accuracy']:.1f}%",
                            'M2_OU_Accuracy': f"{data['method2']['ou_accuracy']:.1f}%",
                            'High_Conf_HDA': f"{data['method1']['high_conf_hda_accuracy']:.1f}%" if data['method1']['high_conf_hda_accuracy'] > 0 else "N/A",
                            'High_AVG_Pattern': f"{data['method1']['high_avg_pattern_accuracy']:.1f}%" if data['method1']['high_avg_pattern_accuracy'] > 0 else "N/A",
                            'High_Conf_Opps': data['method1']['high_conf_opportunities'],
                            'High_AVG_Opps': data['method1']['high_avg_opportunities']
                        })
                    detailed_df = pd.DataFrame(detailed_data)
                    st.dataframe(detailed_df, use_container_width=True)
                else:
                    st.error("❌ Analysis failed. Please check your data and try again.")
    with tab2:
        st.markdown("## 📊 Comprehensive Accuracy Charts")
        if 'score_range_results' in st.session_state:
            results = st.session_state.score_range_results
            # Main accuracy chart
            st.markdown("### 🚀 Accuracy Across All Score Ranges")
            accuracy_chart = create_score_range_accuracy_chart(results)
            st.plotly_chart(accuracy_chart, use_container_width=True)
            # Revolutionary patterns chart
            st.markdown("### ⚡ Revolutionary Patterns Performance")
            patterns_chart = create_revolutionary_patterns_by_range_chart(results)
            st.plotly_chart(patterns_chart, use_container_width=True)
            # Individual method comparisons
            st.markdown("### 🔥 Method Performance Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### METHOD 1 vs METHOD 2: H/D/A")
                score_counts = list(results.keys())
                m1_hda = [results[sc]['method1']['hda_accuracy'] for sc in score_counts]
                m2_hda = [results[sc]['method2']['hda_accuracy'] for sc in score_counts]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=score_counts, y=m1_hda,
                    mode='lines+markers',
                    name='METHOD 1',
                    line=dict(color='#ff6b6b', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=score_counts, y=m2_hda,
                    mode='lines+markers',
                    name='METHOD 2',
                    line=dict(color='#4ecdc4', width=3)
                ))
                fig.update_layout(
                    title='H/D/A Accuracy Comparison',
                    xaxis_title='Number of Scores',
                    yaxis_title='Accuracy (%)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("#### METHOD 1 vs METHOD 2: O/U")
                m1_ou = [results[sc]['method1']['ou_accuracy'] for sc in score_counts]
                m2_ou = [results[sc]['method2']['ou_accuracy'] for sc in score_counts]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=score_counts, y=m1_ou,
                    mode='lines+markers',
                    name='METHOD 1',
                    line=dict(color='#ff6b6b', width=3, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=score_counts, y=m2_ou,
                    mode='lines+markers',
                    name='METHOD 2',
                    line=dict(color='#4ecdc4', width=3, dash='dash')
                ))
                fig.update_layout(
                    title='O/U Accuracy Comparison',
                    xaxis_title='Number of Scores',
                    yaxis_title='Accuracy (%)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 Please run the analysis first to see the charts.")
    with tab3:
        st.markdown("## 🎯 Optimal Score Count Results")
        if 'score_range_results' in st.session_state:
            results = st.session_state.score_range_results
            # Find optimal results
            optimal_results = analyzer.find_optimal_score_counts(results)
            # Display optimal results
            st.markdown("### 🏆 Optimal Score Counts for Maximum Accuracy")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="optimal-card">
                <h3>🔥 METHOD 1 H/D/A</h3>
                <h2>{optimal_results['method1_hda']['best_score_count']} scores</h2>
                <p><strong>Max Accuracy:</strong> {optimal_results['method1_hda']['best_accuracy']:.1f}%</p>
                <p><strong>Improvement:</strong> vs 5 scores baseline</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="optimal-card">
                <h3>⚽ METHOD 1 O/U</h3>
                <h2>{optimal_results['method1_ou']['best_score_count']} scores</h2>
                <p><strong>Max Accuracy:</strong> {optimal_results['method1_ou']['best_accuracy']:.1f}%</p>
                <p><strong>Improvement:</strong> vs 5 scores baseline</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="optimal-card">
                <h3>🧠 METHOD 2 H/D/A</h3>
                <h2>{optimal_results['method2_hda']['best_score_count']} scores</h2>
                <p><strong>Max Accuracy:</strong> {optimal_results['method2_hda']['best_accuracy']:.1f}%</p>
                <p><strong>Improvement:</strong> vs 5 scores baseline</p>
                </div>
                """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="optimal-card">
                <h3>📊 METHOD 2 O/U</h3>
                <h2>{optimal_results['method2_ou']['best_score_count']} scores</h2>
                <p><strong>Max Accuracy:</strong> {optimal_results['method2_ou']['best_accuracy']:.1f}%</p>
                <p><strong>Improvement:</strong> vs 5 scores baseline</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="revolutionary-card">
                <h3>⚡ HIGH CONF H/D/A</h3>
                <h2>{optimal_results['high_conf_hda']['best_score_count']} scores</h2>
                <p><strong>Max Accuracy:</strong> {optimal_results['high_conf_hda']['best_accuracy']:.1f}%</p>
                <p><strong>Revolutionary:</strong> 63.4% target</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="revolutionary-card">
                <h3>🎯 HIGH AVG PATTERN</h3>
                <h2>{optimal_results['high_avg_pattern']['best_score_count']} scores</h2>
                <p><strong>Max Accuracy:</strong> {optimal_results['high_avg_pattern']['best_accuracy']:.1f}%</p>
                <p><strong>Revolutionary:</strong> 75% target</p>
                </div>
                """, unsafe_allow_html=True)
            # Optimal results summary chart
            st.markdown("### 📊 Optimal Score Count Summary")
            optimal_chart = create_optimal_score_count_summary(optimal_results)
            st.plotly_chart(optimal_chart, use_container_width=True)
            # Recommendations
            st.markdown("### 💡 Implementation Recommendations")
            st.markdown(f"""
            <div class="analysis-card">
            <h3>🚀 OPTIMAL IMPLEMENTATION STRATEGY</h3>
            <p><strong>For H/D/A Predictions:</strong></p>
            <ul>
            <li>Use METHOD 1 with {optimal_results['method1_hda']['best_score_count']} closest scores for general predictions</li>
            <li>When high confidence (≥60%), use {optimal_results['high_conf_hda']['best_score_count']} scores for {optimal_results['high_conf_hda']['best_accuracy']:.1f}% accuracy</li>
            </ul>
            <p><strong>For O/U Predictions:</strong></p>
            <ul>
            <li>Use METHOD 1 with {optimal_results['method1_ou']['best_score_count']} scores for general predictions</li>
            <li>When high AVG pattern (≥2.5), use {optimal_results['high_avg_pattern']['best_score_count']} scores for {optimal_results['high_avg_pattern']['best_accuracy']:.1f}% accuracy</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔍 Please run the analysis first to see optimal results.")
    with tab4:
        st.markdown("## 📈 Stability Analysis")
        if 'score_range_results' in st.session_state:
            results = st.session_state.score_range_results
            # Run stability analysis
            stability_analysis = analyzer.analyze_score_range_stability(results)
            st.markdown("### 📊 Stability Metrics Across Score Ranges")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stability-card">
                <h3>🔥 METHOD 1 Stability</h3>
                <p><strong>H/D/A Variance:</strong> {stability_analysis['method1_hda']['variance']:.2f}</p>
                <p><strong>H/D/A Range:</strong> {stability_analysis['method1_hda']['range']:.1f}%</p>
                <p><strong>O/U Variance:</strong> {stability_analysis['method1_ou']['variance']:.2f}</p>
                <p><strong>O/U Range:</strong> {stability_analysis['method1_ou']['range']:.1f}%</p>
                <p><em>Lower variance = more stable performance</em></p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stability-card">
                <h3>🧠 METHOD 2 Stability</h3>
                <p><strong>H/D/A Variance:</strong> {stability_analysis['method2_hda']['variance']:.2f}</p>
                <p><strong>H/D/A Range:</strong> {stability_analysis['method2_hda']['range']:.1f}%</p>
                <p><strong>O/U Variance:</strong> {stability_analysis['method2_ou']['variance']:.2f}</p>
                <p><strong>O/U Range:</strong> {stability_analysis['method2_ou']['range']:.1f}%</p>
                <p><em>Lower variance = more stable performance</em></p>
                </div>
                """, unsafe_allow_html=True)
            # Stability recommendations
            st.markdown("### 🎯 Stability-Based Recommendations")
            # Find most stable method for each prediction type
            most_stable_hda = "METHOD 1" if stability_analysis['method1_hda']['variance'] < stability_analysis['method2_hda']['variance'] else "METHOD 2"
            most_stable_ou = "METHOD 1" if stability_analysis['method1_ou']['variance'] < stability_analysis['method2_ou']['variance'] else "METHOD 2"
            st.markdown(f"""
            <div class="analysis-card">
            <h3>📊 STABILITY RECOMMENDATIONS</h3>
            <p><strong>Most Stable for H/D/A:</strong> {most_stable_hda}</p>
            <p><strong>Most Stable for O/U:</strong> {most_stable_ou}</p>
            <br>
            <p><strong>Key Insights:</strong></p>
            <ul>
            <li>Lower variance indicates more consistent performance across different score ranges</li>
            <li>Stable methods are less sensitive to score count changes</li>
            <li>High accuracy + low variance = optimal combination</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            # Performance vs Stability scatter plot
            st.markdown("### 🎨 Performance vs Stability Visualization")
            score_counts = list(results.keys())
            fig = go.Figure()
            # METHOD 1 H/D/A points
            m1_hda_acc = [results[sc]['method1']['hda_accuracy'] for sc in score_counts]
            m1_hda_var = [abs(acc - np.mean(m1_hda_acc)) for acc in m1_hda_acc]  # Deviation from mean as stability measure
            fig.add_trace(go.Scatter(
                x=m1_hda_var,
                y=m1_hda_acc,
                mode='markers',
                marker=dict(color='#ff6b6b', size=10),
                name='METHOD 1 H/D/A',
                text=[f"{sc} scores" for sc in score_counts],
                hovertemplate='%{text}<br>Stability: %{x:.2f}<br>Accuracy: %{y:.1f}%<extra></extra>'
            ))
            # METHOD 2 H/D/A points
            m2_hda_acc = [results[sc]['method2']['hda_accuracy'] for sc in score_counts]
            m2_hda_var = [abs(acc - np.mean(m2_hda_acc)) for acc in m2_hda_acc]
            fig.add_trace(go.Scatter(
                x=m2_hda_var,
                y=m2_hda_acc,
                mode='markers',
                marker=dict(color='#4ecdc4', size=10),
                name='METHOD 2 H/D/A',
                text=[f"{sc} scores" for sc in score_counts],
                hovertemplate='%{text}<br>Stability: %{x:.2f}<br>Accuracy: %{y:.1f}%<extra></extra>'
            ))
            fig.update_layout(
                title='🎨 Performance vs Stability (H/D/A)',
                xaxis_title='Deviation from Mean (Lower = More Stable)',
                yaxis_title='Accuracy (%)',
                height=500
            )
            # Add ideal zone
            fig.add_annotation(
                x=min(m1_hda_var + m2_hda_var) * 0.1,
                y=max(m1_hda_acc + m2_hda_acc) * 0.95,
                text="🎯 IDEAL ZONE<br>(High Accuracy + Low Deviation)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                font=dict(color="green", size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 Please run the analysis first to see stability metrics.")

if __name__ == "__main__":
    main()