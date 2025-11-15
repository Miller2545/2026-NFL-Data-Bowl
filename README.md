# 2026 NFL Data Bowl Prediction Competition

## Overview

This will be the location for files and whatnot for team NFL YIPPEE (name still a wip) to collaborate.



### Git Basics: Cloning, Branching, Saving, and Pushing Your Work

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

