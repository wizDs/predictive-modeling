# setup 
``` bash
docker pull mcr.microsoft.com/mssql/server:2022-latest
docker run --name sql-server --hostname sql-server -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=StrongPassword123" -p 1433:1433 -d mcr.microsoft.com/mssql/server:2022-latest 
docker exec -it sql-server bash
/opt/mssql-tools/bin/sqlcmd -S localhost -U SA -P "StrongPassword123"
```

``` bash
CREATE DATABASE Electricity;
GO;
```


# bash
``` bash
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --requirement requirements.txt 
python3 -m pip install --editable .
export DB_CONNECTION="mssql+pyodbc://[username]:[password]@[server]:[port]/[database]?driver=[driver]"
python3 scripts/run_etl.py
```

example on `DB_CONNECTION="mssql+pyodbc://SA:StrongPassword123@localhost:1433/Electricity?driver=SQL+SERVER"`

# cmd
``` cmd
python -m pip install --upgrade pip
python -m pip install --upgrade build
python -m pip install --requirement requirements.txt 
python -m pip install --editable .
setx DB_CONNECTION "mssql+pyodbc://[username]:[password]@[server]:[port]/[database]?driver=[driver]"
python scripts/run_etl.py
```

- https://hub.docker.com/_/microsoft-mssql-server
- https://docs.microsoft.com/en-us/sql/linux/quickstart-install-connect-docker?view=sql-server-ver16&pivots=cs1-bash