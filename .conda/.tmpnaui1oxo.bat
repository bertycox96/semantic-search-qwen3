@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\vyart\anaconda3\condabin\conda.bat" activate "c:\Users\vyart\Downloads\semantic-search-qwen3\semantic-search-qwen3\.conda"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@c:\Users\vyart\Downloads\semantic-search-qwen3\semantic-search-qwen3\.conda\python.exe -Wi -m compileall -q -l -i C:\Users\vyart\AppData\Local\Temp\tmpw4w1tpwq -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
