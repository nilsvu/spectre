#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse as ap
import codecs
import json
import logging
import requests


def upload(api_token, json_file_name, zip_file_name, upload_to_sandbox):
    base_url = ("https://sandbox.zenodo.org"
                if upload_to_sandbox else "https://zenodo.org")
    if base_url != "https://sandbox.zenodo.org":
        print("Not yet allowed to upload to Zenodo. We are still "
              "debugging, sorry!")
        return

    # Read the .zenodo.json file and wrap it into the "metadata" element of what
    # we send to Zenodo.
    metadata = json.dumps(json.loads(
        codecs.open(json_file_name, 'r', 'utf-8').read()),
                          ensure_ascii=True)
    metadata = "{\"metadata\": " + metadata + "}"

    headers = {"Content-Type": "application/json"}
    params = {'access_token': api_token}
    url = f"{base_url}/api/deposit/depositions"

    # Try to create the entry on Zenodo
    response = requests.post(url,
                             params=params,
                             data=metadata,
                             headers=headers)
    if response.status_code > 210:
        print("Error happened during submission, status code: " +
              str(response.status_code))
        print(response.text)
        return

    # Upload the actual zip file we got from GitHub
    bucket_url = response.json()["links"]["bucket"]
    with open(zip_file_name, "rb") as zip_file:
        zip_name = "spectre.zip"  #TODO: we need to update the zip name!
        upload_response = requests.put(f"{bucket_url}/{zip_name}",
                                       data=zip_file,
                                       params=params)

    print(
        "\nUploaded SpECTRE to {}\nPlease note that you must still double check"
        " that the info is accurate and then publish the code for the DOI to "
        "be active.".format(response.json()["links"]["html"]))


def parse_args():
    """
    Parse the command line arguments
    """
    parser = ap.ArgumentParser(
        description="Upload SpECTRE zip archive of release to Zenodo.",
        formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--api-token',
                        type=str,
                        required=True,
                        help="The Zenodo API token")
    parser.add_argument('--zenodo-json',
                        type=str,
                        required=True,
                        help="The Zenodo JSON file. Usually name .zenodo.json")
    parser.add_argument('--zip-file',
                        type=str,
                        required=True,
                        help="The ZIP file of the SpECTRE release to upload.")
    sandbox_group = parser.add_mutually_exclusive_group()
    sandbox_group.add_argument('--zenodo-sandbox',
                               dest='zenodo_sandbox',
                               action='store_true')
    sandbox_group.add_argument('--no-zenodo-sandbox',
                               dest='zenodo_sandbox',
                               action='store_false')
    sandbox_group.set_defaults(zenodo_sandbox=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_args = vars(parse_args())
    upload(input_args['api_token'], input_args['zenodo_json'],
           input_args['zip_file'], input_args['zenodo_sandbox'])
