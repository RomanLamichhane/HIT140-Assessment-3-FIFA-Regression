from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    'build_datasets.py',
    'Raman_Regression_2_1_Goal_Difference/code/raman_regression_2_1.py',
    'Saroj_Regression_2_2_Goals_Scored/code/saroj_regression_2_2.py',
    'Abhinav_Data_Validation_Model_Comparison/code/abhinav_validate_datasets.py',
    'Abhinav_Data_Validation_Model_Comparison/code/abhinav_model_comparison.py',
    'Raman_Regression_2_1_Goal_Difference/code/raman_regression_2_1_enhanced.py',
    'Saroj_Regression_2_2_Goals_Scored/code/saroj_regression_2_2_enhanced.py',
    'Abhinav_Data_Validation_Model_Comparison/code/abhinav_enhanced_model_comparison.py',
]

for rel in SCRIPTS:
    print(f'\n>>> Running {rel}')
    subprocess.run([sys.executable, str(ROOT / rel)], cwd=ROOT, check=True)

print('\nSUCCESS - all HIT140 Assessment 3 scripts completed.')
