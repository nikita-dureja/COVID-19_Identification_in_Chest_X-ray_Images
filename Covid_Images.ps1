cd "C:\Users\Spare\OneDrive - Swinburne University\Semester 2\Data Science Project 1\Project\Data Merging\COVID-19"
dir | rename-item -NewName {$_.name -replace " ",""}