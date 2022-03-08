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
- Setting up an ssh key: To push changes to GitHub you will need an ssh key pair that will be used to authenticate your account when you make changes.
- Cloning the repository: When first downloading this repo, do
  ```
  $ git clone git@github.com:MikeSoughton/twitter-nlp.git
  ```
  You should only need to do this once.
