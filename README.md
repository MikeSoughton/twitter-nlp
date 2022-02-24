# twitter-nlp

### Running in Google Colab

We can run this in [Google Colab](https://colab.research.google.com/) by going to GitHub on that page and (it needs to be tested that the GitHub accounts and repos sync up correctly) entering your name, remembering to tick 'Include private repos'. Next select this repo (and the appropriate branch) and then the notebook to be opened. The notebook can be run and edited in Colab and then saved to this GitHub under File->Save a copy in GitHub. You can also save it to your drive as well if you wish. 

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

ON FURTHER TESTING THERE ARE FURTHER ISSUES WITH RUNNING SNSCRAPE ON COLAB. I HAVE NO PROBLEM RUNNING THEM LOCALLY IF THIS REMAINS A PROBLEM I WAS ONLY LOOKING INTO THIS FOR THE SAKE OF CONSISTENCY.

All other notebooks which do not require snscrape we should be able to run normally.
