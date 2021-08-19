cd "C:\Users\Spare\OneDrive - Swinburne University\Semester 2\Data Science Project 1\Project\Data Merging\NORMAL"
dir | rename-item -NewName {$_.name -replace "\)",""}
dir | rename-item -NewName {$_.name -replace " \(","-"}