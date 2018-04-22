SVM = [0.79536, 0.06331, 0.07196, 0.31411, 0.84649, 0.2666, 0.37149, 0.54253, 0.51166, 0.23517, 0.14401, 0.47779, 0.21405, 0.37301, 0.04931, 0.77296, 0.2484, 0.37301, 0.13322, 0.52669, 0.275898, 0.11593, 0.16299, 0.10455, 0.01365, 0.40576, 0.58024]
ARIMA = [0.085, 0.699, 0.085, 0.079, 0.353, 1.870, 0.412, 0.468, 0.270, 1.536, 0.226, 0.149, 0.517, 0.254, 0.467, 0.137, 0.495, 0.303, 0.219, 0.741, 0.361, 0.310, 0.176, 0.055, 0.043, 0.682, 0.594]
benchmark = [0.342, 0.612, 0.662, 0.643, 0.452, 1.550, 0.375, 0.623, 0.531, 0.457, 0.385, 0.331, 0.292, 0.113, 0.453, 0.272, 0.241, 0.503, 0.591, 0.530, 0.732, 1.012, 0.496, 0.235, 0.282, 0.429, 1.254]

print(len(SVM))
print(len(ARIMA))
print(len(benchmark))
from scipy.stats import wilcoxon, mannwhitneyu, ttest_ind

print('svm arima')
print(wilcoxon(SVM,ARIMA))
print('arima benchmark')
print(wilcoxon(ARIMA,benchmark))
print('svm benchmark')
print(wilcoxon(SVM,benchmark))

print('svm arima')
print(mannwhitneyu(SVM,ARIMA))
print('arima benchmark')
print(mannwhitneyu(ARIMA,benchmark))
print('svm benchmark')
print(mannwhitneyu(SVM,benchmark))

print('svm arima')
print(ttest_ind(SVM,ARIMA))
print('arima benchmark')
print(ttest_ind(ARIMA,benchmark))
print('svm benchmark')
print(ttest_ind(SVM,benchmark))
