
# Show column names
```sqlite3
.headers on
.mode column
```

# Show metadata info
```sqlite3
.schema
.table
```


# Get column metadata
```sql
SELECT 
  m.name as table_name, 
  p.name as column_name,
  p.type as column_type
FROM 
  sqlite_master AS m
JOIN 
  pragma_table_info(m.name) AS p
WHERE
  m.type = 'table' 
ORDER BY 
  m.name, 
  p.cid
```

```sql
SELECT 
  m.name as table_name, 
  p.name as column_name,
  p.type as column_type
FROM 
  sqlite_master AS m
JOIN 
  pragma_table_info(m.name) AS p
WHERE
  m.type = 'table' 
  and m.name = 'hospital_general_information'
ORDER BY 
  m.name, 
  p.cid
```
