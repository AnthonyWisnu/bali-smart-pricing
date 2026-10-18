@echo off
title Bali Smart Pricing Engine - Web Dashboard
color 0B
cls
echo ===============================================================================
echo        BALI SMART VILLA ^& HOMESTAY PRICING ENGINE - WEB DASHBOARD
echo                      Menyalakan server Streamlit...
echo ===============================================================================
echo.
echo Membuka peramban di http://localhost:8501...
echo Tekan Ctrl+C di terminal ini jika ingin menghentikan aplikasi.
echo.
streamlit run app.py
pause
