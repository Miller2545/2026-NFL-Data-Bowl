# 2026 NFL Data Bowl Prediction Competition

## Overview

This will be the location for files and whatnot for team NFL YIPPEE (name still a wip) to collaborate.



### Git Tutorial

This quick guide will teach you how to:

- Clone the main repository  
- Create your own branch  
- Save (commit) your changes  
- Push your branch to the remote repository  

No prior Git knowledge required.

---

#### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Miller2545/2026-NFL-Data-Bowl.git

cd 2026-NFL-Data-Bowl
```

(Note: The above is if you are working in linux, for windows it will just create the folder 2026-NFL-Data-Bowl and you can access the work from there!)

#### 2. Creating Your Working Branch

```bash
git checkout main
git pull
```

The above ensures you are on our main branch and pulls any changes from github.

```bash
git checkout -b <gmuid>
```

This will create your own branch so that we have a main branch that we will always have something working code-wise, and our own branches that we change ourselves. With us all working in the same project this allows us to be able to work within the same files and not step on eachothers toes.


#### 3. Gitting Committed
```bash
git status
```

This will show you if there were any changes made on your current branch that need to be committed and pushed. It should look something like:

```bash
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

If you haven't saved a file with changes yet.

When you save changes it will look like the following:

```bash
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.md

no changes added to commit (use "git add" and/or "git commit -a")
```

Now since we have changes, we can either use the UI in either RStudio or VScode, or go to the command line and add, commit, and push our changes.

```bash
git add .
git commit -m "Added: git commit to tutorial section of README.md"
git push
```

#### 4. Profit

The following above will add all files to your commit, commit the changes with the message "Added: git commit to tutorial section of README.md", and then push the changes to github.

This will push it to your working branch. When it comes to merging all of our work into the main working branch, I can handle that to make is easier!
