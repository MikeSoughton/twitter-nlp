# twitter-nlp

### Running in Google Colab

We can run this in [Google Colab](https://colab.research.google.com/) by going to GitHub on that page and (it needs to be tested that the GitHub accounts and repos sync up correctly) entering your name, remembering to tick 'Include private repos'. Next select this repo (and the appropriate branch) and then the notebook to be opened. The notebook can be run and edited in Colab and then saved to this GitHub under File->Save a copy in GitHub. You can also save it to your drive as well if you wish.

#### Running in Google Colab with local runtime (not working)

There are still issues running snscrape on Google Colab even using a local runtime. I'll leave these instructions for getting a local runtime working here for now, since they do still work, it is just the snscrape-colab interaction which fails.

The extraction notebook requires the dev version [snscrape](https://github.com/JustAnotherArchivist/snscrape) to be installed, however this will not work in the Python3.7 that Colab currently (24/02/2022) runs on as it needs Python3.8 or higher. We can of course just run this on our own laptops without Colab, but if we wish to use Colab then as a workaround we can connect to our own local runtimes. To allow this, follow the steps in [https://research.google.com/colaboratory/local-runtimes.html](https://research.google.com/colaboratory/local-runtimes.html) and do in your terminal (one time to install):
  ```
  $ pip install jupyter_http_over_ws
  $ jupyter serverextension enable --py jupyter_http_over_ws
  ```
Then whenever you want to connect to the local runtime do in your terminal:
  ```
  $ jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
  ```
Copy the url link it provides and paste it into the box under Connect->Connect to a local runtime, in the right hand corner of a Colab notebook.

On further testing there are further issues with running snscrape on colab. I have no problem running them locally if this remains a problem I was only looking into this for the sake of consistency.

All other notebooks which do not require snscrape we should be able to run normally.

### Using git

Here are some instructions for using git to work on the project on your computer and then push changes back to the GitHub. I have not actually tried working on the same project with multiple people at the same time so the cloning, pulling and merging may take some experimenting to get right. From what I hear it *should* be fairly straightforward as git is designed to handle multiple people working on the same file simultaneously but if there are any problems we can learn from them. How much we use git commands will depend on how much of the project we run locally.

Here are the basic instructions that we should need:

#### Setting up ssh keys, cloning and changing branches

- Setting up an ssh key: To push changes to GitHub you will need an ssh key pair that will be used to authenticate your account when you make changes. You can generate a private key that will sit in your computer and a public key that you upload to GitHub. When you run the git push command GitHub will the public key against your private key and allow you in. To generate the key pair do:
  ```
  $ ssh-keygen -t rsa -b 4096 -C "<your github email address>"
  ```
  It will give you the option of what to call the file and where to save it and also the option of making a passphrase for it (you can just press enter to leave it empty). Next activate your computer's ssh agent with
  ```
  $ eval $(ssh-agent -s)
  ```
  Now add your private ssh key to the agent with
  ```
  $ ssh-add <path to private key>
  ```
  with the default path as `~/.ssh/id_rsa`, and enter the passphrase if you made one. Finally copy the contents of the public key (default path `~/.ssh/id_rsa.pub`), head to your GitHub &rarr; Settings &rarr; SSH and GPG keys &rarr; New SSH key, make a title that makes sense to you, copy the contents of the public key and click Add SSH key. 
- Cloning the repository: When first downloading this repo, do
  ```
  $ git clone git@github.com:MikeSoughton/twitter-nlp.git
  ```
  You should only need to do this once.
- Changing to a different branch: You can work on different 'branches' with the idea being  that each branch corresponds to a different workflow. For now we will just have a main and a dev branch. To change between branches do
  ```
  $ git checkout <branch name>
  ```
  You won't be able to change branch if there are uncommitted changes on your current branch.

#### The standard commands to add, commit and push

- Adding files to be ready to be 'committed': Add a single file with
  ```
  $ git add <file name>
  ```
  or add all files with
  ```
  $ git add -A
  ```
  or
  ```
  $ git add .
  ```
- Committing files: 'Commit' changes, **ready** to be pushed to GitHub with
  ```
  $ git commit -m "Your message describing your changes"
  ```
- Pushing changes to GitHub: Push these committed changes to GitHub without
  ```
  $ git push orign <branch name>
  ```
  Currently I am pushing to the branch dev but this could be any branch.

Note that whilst I would always do the git add and git commit steps before pushing, I was investigating how to update your local branch with any changes made to the remote branch (the one on GitHub). The command for this is (note that there are alternatives to this that I do not yet fully understand, see for example `$ git fetch` [https://www.atlassian.com/git/tutorials/syncing/git-fetch](https://www.atlassian.com/git/tutorials/syncing/git-fetch))
```
$ git pull --rebase origin <branch name>
```
where at the moment the branch name I used was dev. There should hopefully not be any issues with this step but if there are checkout [https://www.atlassian.com/git/tutorials/comparing-workflows](https://www.atlassian.com/git/tutorials/comparing-workflows). The funny thing that I noticed was that after doing this, when I tried adding and committing I was told 'nothing to commit, working tree clean' even though I had made changes locally. This *may* have been because I had previously added and committed changes, but I don't think I did so it's a bit odd. All I had to do was push with `$ git push orign <branch name>`.

Okay; so when I came to add, commit and push new changes (without pulling again) I **did** have to add and commit before pushing, so maybe I had already done them before pulling. Or maybe pulling is just weird - this we can figure out in the future.

So once we are up and running, the four commands that we would use are pull, add, commit and push.

#### Merging
This is a bit more advanced and is done when... well it's a bit complicated and we won't use it now, but maybe we will use it later.

#### Undo commands

For when you mistakenly add, commit or push and want to undo them:
- How to undo git add:
- How to undo git commit:
- How to undo git push:
