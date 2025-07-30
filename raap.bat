@ECHO off

IF "%1"=="" (
    ECHO No first argument provided
    GOTO :END
)

IF "%1"=="run" (
    GOTO :RUN
)

ECHO Invalid or unimplemented command: %1%
GOTO :END

:RUN

if "%2"=="" (
    ECHO No command to run provided
    GOTO :END
)

call python run_with_full_config.py %2%
GOTO :END


:END
@ECHO on
