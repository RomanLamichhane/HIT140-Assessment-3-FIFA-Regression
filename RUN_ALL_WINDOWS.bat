@echo off
setlocal

echo ============================================================
echo HIT140 Assessment 3 - Full Reproducible Run
echo ============================================================

python build_datasets.py || goto :error
python Raman_Regression_2_1_Goal_Difference\code\raman_regression_2_1.py || goto :error
python Saroj_Regression_2_2_Goals_Scored\code\saroj_regression_2_2.py || goto :error
python Abhinav_Data_Validation_Model_Comparison\code\abhinav_validate_datasets.py || goto :error
python Abhinav_Data_Validation_Model_Comparison\code\abhinav_model_comparison.py || goto :error
python Raman_Regression_2_1_Goal_Difference\code\raman_regression_2_1_enhanced.py || goto :error
python Saroj_Regression_2_2_Goals_Scored\code\saroj_regression_2_2_enhanced.py || goto :error
python Abhinav_Data_Validation_Model_Comparison\code\abhinav_enhanced_model_comparison.py || goto :error

echo.
echo SUCCESS - all scripts completed.
pause
exit /b 0

:error
echo.
echo ERROR - one of the scripts failed. Review the message above.
pause
exit /b 1
