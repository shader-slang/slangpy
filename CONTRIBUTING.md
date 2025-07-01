# SlangPy Open Source Project

## Contribution Guide

Thank you for considering contributing to the SlangPy project! We welcome your help to improve and enhance our project. Please take a moment to read through this guide to understand how you can contribute.

This document is designed to guide you in contributing to the project. It is intended to be easy to follow without sending readers to other pages and links. You can simply copy and paste the command lines described in this document.

* Contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant the rights to use your contribution.
* When you submit a pull request, a CLA bot will determine whether you need to sign a CLA. Simply follow the instructions provided.
* Please read and follow the contributor [Code of Conduct](CODE_OF_CONDUCT).
* Bug reports and feature requests should be submitted via the GitHub issue tracker.
* Changes should ideally come in as small pull requests on top of master, coming from your own personal fork of the project.
* Large features that will involve multiple contributors or a long development time should be discussed in issues and broken down into smaller pieces that can be implemented and checked in stages.

## Table of Contents
1. [Contribution Process](#contribution-process)
   - [Forking the Repository](#forking-the-repository)
   - [Cloning Your Fork](#cloning-your-fork)
   - [Creating a Branch](#creating-a-branch)
   - [Build SlangPy from Source](#build-slangpy-from-source)
   - [Making and Testing Changes](#making-and-testing-changes)
   - [Commit to the Branch](#commit-to-the-branch)
   - [Push to Forked Repository](#push-to-forked-repository)
2. [Pull Request](#pull-request)
   - [Addressing Code Reviews](#addressing-code-reviews)
3. [Issue Tracking](#issue-tracking)
4. [Communication](#communication)
5. [License](#license)

## Contribution Process

### Forking the Repository
Navigate to the [SlangPy repository](https://github.com/shader-slang/slangpy).
Click on the "Fork" button in the top right corner to create a copy of the repository in your GitHub account.
This document will assume that the name of your forked repository is "slangpy".
Make sure your "Actions" are enabled. Visit your forked repository, click on the "Actions" tab, and enable the actions.

### Cloning Your Fork
1. Clone your fork locally, replacing "USER-NAME" in the command below with your actual username.
   ```
   $ git clone --recursive --tags https://github.com/USER-NAME/slangpy.git
   $ cd slangpy
   ```

2. Fetch tags by adding the original repository as an upstream.
   It is important to have tags in your forked repository because our workflow/action uses the information for the build process. But the tags are not fetched by default when you fork a repository in GitHub. You need to add the original repository as an upstream and fetch tags manually.
   ```
   $ git remote add upstream https://github.com/shader-slang/slangpy.git
   $ git fetch --tags upstream
   ```

   You can check whether the tags are fetched properly with the following command.
   ```
   $ git tag -l
   ```

3. Push tags to your forked repository.
   The tags are fetched to your local machine but haven't been pushed to the forked repository yet. You need to push tags to your forked repository with the following command.
   ```
   $ git push --tags origin
   ```

### Creating a Branch
Create a new branch for your contribution:
```
$ git checkout -b feature/your-feature-name
```

### Build SlangPy from Source
Please follow the instructions in the ["Compiling" page of the Developer Guide](https://slangpy.shader-slang.org/en/latest/src/developer_guide/compiling.html).

### Making and Testing Changes
After making your changes, test your changes thoroughly to ensure they do not introduce new issues. This is done by running unit tests via `pytest` from the repository root directory. For more details about running `pytest`, please refer to the ["Testing" page of the Developer Guide](https://slangpy.shader-slang.org/en/latest/src/developer_guide/testing.html).

If you are familiar with Workflows/Actions in GitHub, you can check [Our Workflows](.github/workflows). The "Run tests" sections in [ci.yml](.github/workflows/ci.yml) is where `pytest` is run by [ci.py](.build_agent/ci.py).

### Commit to the Branch
Commit your changes to the branch with a descriptive commit message:
```
$ git commit
```

It is important to have a descriptive commit message. Unlike comments inside the source code, the commit messages don't spoil over time because they are tied to specific changes and can be reviewed by many people many years later.

Here is a good example of a commit message:

> Add user authentication feature
>
> Fixes #1234
>
> This commit introduces a new user authentication feature. It includes changes to the login page, user database, and session management to provide secure user authentication.

### Push to Forked Repository
Push your branch to your forked repository with the following command:
```
$ git push origin feature/your-feature-name
```

After the changes are pushed to your forked repository, the change needs to be merged to the final destination `shader-slang/slangpy`.
In order to proceed, you will need to create a "Pull Request," or "PR" for short.

When you push to your forked repository, `git-push` usually prints a URL that allows you to create a PR.

If you missed a chance to use the URL, you can still create a PR from the GitHub webpage.
Go to your forked repository and change the branch name to the one you used for `git-push`.
It will show a message like "This branch is 1 commit ahead of `shader-slang/slangpy:main`."
You can create a PR by clicking on the message.

## Pull Request
Once a PR is created against `shader-slang/slangpy:main`, the PR will be merged when the following conditions are met:
1. The PR is reviewed and got approval.
1. All of the workflows pass.

When the conditions above are all met, you will have a chance to rewrite the commit message. Since the SlangPy repo uses the "squash" strategy for merging, multiple commits in your PR will become one commit. By default, GitHub will concatenate all of the commit messages sequentially, but often it is not readable. Please rewrite the final commit message in a way that people can easily understand what the purpose of the commit is.

### Addressing Code Reviews
After your pull request is created, you will receive code reviews from the community within 24 hours.

The PR requires approval from people who have permissions. They will review the changes before approving the pull. During this step, you will get feedback from other people, and they may request you to make some changes.

Follow-up changes that address review comments should be pushed to your pull request branch as additional commits. Any additional commits made to the same branch in your forked repository will show up on the PR as incremental changes.

When your branch is out of sync with top-of-tree, submit a merge commit to keep them in sync. Do not rebase and force push after the PR is created to keep the change history during the review process.

Use these commands to sync your branch:
```
$ git fetch upstream master
$ git merge upstream/master # resolve any conflicts here
```

The SlangPy repository uses the squash strategy for merging pull requests, which means all your commits will be squashed into one commit by GitHub upon merge.

## Issue Tracking
We track all our work with GitHub issues. Check the [Issues](https://github.com/shader-slang/slangpy/issues) for open issues. If you find a bug or want to suggest an enhancement, please open a new issue.

## Communication
Join our [Discussions](https://github.com/shader-slang/slang/discussions),
and our [Discord Server](https://khr.io/slangdiscord).

## License
By contributing to SlangPy, you agree that your contributions will be licensed under the Apache License 2.0 with LLVM Exception. The full text of the License can be found in the [LICENSE](LICENSE.txt) file in the root of the repository.
