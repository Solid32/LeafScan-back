{"payload":{"allShortcutsEnabled":true,"fileTree":{"leafscan":{"items":[{"name":"__init__.py","path":"leafscan/__init__.py","contentType":"file"},{"name":"main.py","path":"leafscan/main.py","contentType":"file"}],"totalCount":2},"":{"items":[{"name":"api","path":"api","contentType":"directory"},{"name":"leafscan","path":"leafscan","contentType":"directory"},{"name":"models","path":"models","contentType":"directory"},{"name":"notebooks","path":"notebooks","contentType":"directory"},{"name":"pckglist","path":"pckglist","contentType":"directory"},{"name":"raw_data","path":"raw_data","contentType":"directory"},{"name":"scripts","path":"scripts","contentType":"directory"},{"name":"tests","path":"tests","contentType":"directory"},{"name":".env.sample","path":".env.sample","contentType":"file"},{"name":".env.yaml.sample","path":".env.yaml.sample","contentType":"file"},{"name":".envrc","path":".envrc","contentType":"file"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":"Dockerfile","path":"Dockerfile","contentType":"file"},{"name":"Makefile","path":"Makefile","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"project_plan.jpg","path":"project_plan.jpg","contentType":"file"},{"name":"project_plan.xlsx","path":"project_plan.xlsx","contentType":"file"},{"name":"requirements.txt","path":"requirements.txt","contentType":"file"},{"name":"requirements_dev.txt","path":"requirements_dev.txt","contentType":"file"},{"name":"setup.py","path":"setup.py","contentType":"file"}],"totalCount":20}},"fileTreeProcessingTime":8.379605,"foldersToFetch":[],"reducedMotionEnabled":"system","repo":{"id":649708764,"defaultBranch":"master","name":"LeafScan-back","ownerLogin":"Solid32","currentUserCanPush":true,"isFork":false,"isEmpty":false,"createdAt":"2023-06-05T15:14:36.000+02:00","ownerAvatar":"https://avatars.githubusercontent.com/u/130365076?v=4","public":true,"private":false},"refInfo":{"name":"master","listCacheKey":"v0:1686136454.114142","canEdit":true,"refType":"branch","currentOid":"d5b1eec2ecf1bf8498f4f0b1aceae921e8f14c87"},"path":"leafscan/main.py","currentUser":{"id":130365076,"login":"Solid32","userEmail":"guillaume.chinzi@hotmail.be"},"blob":{"rawBlob":"import numpy as np\nimport os\nimport PIL\nimport PIL.Image\nimport tensorflow as tf\nimport tensorflow_datasets as tfds\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n\n\n#comment data_dir first time you use this cell\ndata_dir = '~/tensorflow_datasets'\n\n(train_ds, val_ds, test_ds), metadata = tfds.load(\n    'plant_village',\n    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],\n    with_info=True,\n    as_supervised=True,\n    data_dir=data_dir\n)\ntrain_ds = train_ds.batch(32)\nval_ds = val_ds.batch(32)\ntest_ds = test_ds.batch(32)\n\n#Needed to be able to use 2 GPU's on google VM\n#<--\nmirrored_strategy = tf.distribute.MirroredStrategy()\nwith mirrored_strategy.scope():\n#-->\n    model = tf.keras.Sequential([\n    tf.keras.layers.Rescaling(1./255, input_shape=(256,256,3)),\n    tf.keras.layers.RandomFlip(\"horizontal_and_vertical\"),\n    tf.keras.layers.CenterCrop(height=224, width=224),\n    tf.keras.layers.RandomRotation(0.2),\n    tf.keras.layers.Conv2D(64, 4, activation='relu'),\n    tf.keras.layers.MaxPooling2D(),\n    tf.keras.layers.Conv2D(32, 3, activation='relu'),\n    tf.keras.layers.Flatten(),\n    tf.keras.layers.Dense(246, activation='relu'),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dropout(0.1),\n    tf.keras.layers.Dense(38, activation='softmax')\n    ])\n\ninitial_learning_rate = 0.0015\nlr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(\n    initial_learning_rate,\n    decay_steps=2000,\n    decay_rate=0.9\n)\noptimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)\n\nmodel.compile(optimizer=optimizer,\n              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n              metrics=['accuracy'])\n\nfrom tensorflow.keras.callbacks import EarlyStopping\nes = EarlyStopping(patience=4, restore_best_weights=True)\n\n\nmodel.compile(optimizer='adam',\n              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n              metrics=['accuracy'])\n\n\nmodel.fit(train_ds,\n          callbacks=es,\n          validation_data=val_ds,\n          epochs=50)\n\nmodel.evaluate(test_ds)\n\n\n# if __name__ == '__main__':\n#     #preprocess()\n#     train()\n#     evaluate()\n#     pred()\n","colorizedLines":null,"stylingDirectives":[[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":15,"cssClass":"pl-k"},{"start":16,"end":18,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":9,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":10,"cssClass":"pl-v"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":10,"cssClass":"pl-v"},{"start":11,"end":16,"cssClass":"pl-v"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":17,"cssClass":"pl-s1"},{"start":18,"end":20,"cssClass":"pl-k"},{"start":21,"end":23,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":26,"cssClass":"pl-s1"},{"start":27,"end":29,"cssClass":"pl-k"},{"start":30,"end":34,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":17,"cssClass":"pl-s1"},{"start":18,"end":24,"cssClass":"pl-s1"},{"start":25,"end":27,"cssClass":"pl-k"},{"start":28,"end":31,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":13,"cssClass":"pl-s1"},{"start":14,"end":16,"cssClass":"pl-k"},{"start":17,"end":19,"cssClass":"pl-s1"}],[],[],[],[{"start":0,"end":46,"cssClass":"pl-c"}],[{"start":0,"end":8,"cssClass":"pl-s1"},{"start":9,"end":10,"cssClass":"pl-c1"},{"start":11,"end":34,"cssClass":"pl-s"}],[],[{"start":1,"end":9,"cssClass":"pl-s1"},{"start":11,"end":17,"cssClass":"pl-s1"},{"start":19,"end":26,"cssClass":"pl-s1"},{"start":29,"end":37,"cssClass":"pl-s1"},{"start":38,"end":39,"cssClass":"pl-c1"},{"start":40,"end":44,"cssClass":"pl-s1"},{"start":45,"end":49,"cssClass":"pl-en"}],[{"start":4,"end":19,"cssClass":"pl-s"}],[{"start":4,"end":9,"cssClass":"pl-s1"},{"start":9,"end":10,"cssClass":"pl-c1"},{"start":11,"end":24,"cssClass":"pl-s"},{"start":26,"end":42,"cssClass":"pl-s"},{"start":44,"end":57,"cssClass":"pl-s"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":13,"end":14,"cssClass":"pl-c1"},{"start":14,"end":18,"cssClass":"pl-c1"}],[{"start":4,"end":17,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":18,"end":22,"cssClass":"pl-c1"}],[{"start":4,"end":12,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":13,"end":21,"cssClass":"pl-s1"}],[],[{"start":0,"end":8,"cssClass":"pl-s1"},{"start":9,"end":10,"cssClass":"pl-c1"},{"start":11,"end":19,"cssClass":"pl-s1"},{"start":20,"end":25,"cssClass":"pl-en"},{"start":26,"end":28,"cssClass":"pl-c1"}],[{"start":0,"end":6,"cssClass":"pl-s1"},{"start":7,"end":8,"cssClass":"pl-c1"},{"start":9,"end":15,"cssClass":"pl-s1"},{"start":16,"end":21,"cssClass":"pl-en"},{"start":22,"end":24,"cssClass":"pl-c1"}],[{"start":0,"end":7,"cssClass":"pl-s1"},{"start":8,"end":9,"cssClass":"pl-c1"},{"start":10,"end":17,"cssClass":"pl-s1"},{"start":18,"end":23,"cssClass":"pl-en"},{"start":24,"end":26,"cssClass":"pl-c1"}],[],[{"start":0,"end":46,"cssClass":"pl-c"}],[{"start":0,"end":4,"cssClass":"pl-c"}],[{"start":0,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":22,"cssClass":"pl-s1"},{"start":23,"end":33,"cssClass":"pl-s1"},{"start":34,"end":50,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":22,"cssClass":"pl-s1"},{"start":23,"end":28,"cssClass":"pl-en"}],[{"start":0,"end":4,"cssClass":"pl-c"}],[{"start":4,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":12,"end":14,"cssClass":"pl-s1"},{"start":15,"end":20,"cssClass":"pl-s1"},{"start":21,"end":31,"cssClass":"pl-v"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":29,"cssClass":"pl-v"},{"start":30,"end":32,"cssClass":"pl-c1"},{"start":32,"end":33,"cssClass":"pl-c1"},{"start":33,"end":36,"cssClass":"pl-c1"},{"start":38,"end":49,"cssClass":"pl-s1"},{"start":49,"end":50,"cssClass":"pl-c1"},{"start":51,"end":54,"cssClass":"pl-c1"},{"start":55,"end":58,"cssClass":"pl-c1"},{"start":59,"end":60,"cssClass":"pl-c1"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":30,"cssClass":"pl-v"},{"start":31,"end":56,"cssClass":"pl-s"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":30,"cssClass":"pl-v"},{"start":31,"end":37,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-c1"},{"start":38,"end":41,"cssClass":"pl-c1"},{"start":43,"end":48,"cssClass":"pl-s1"},{"start":48,"end":49,"cssClass":"pl-c1"},{"start":49,"end":52,"cssClass":"pl-c1"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":34,"cssClass":"pl-v"},{"start":35,"end":38,"cssClass":"pl-c1"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":26,"cssClass":"pl-v"},{"start":27,"end":29,"cssClass":"pl-c1"},{"start":31,"end":32,"cssClass":"pl-c1"},{"start":34,"end":44,"cssClass":"pl-s1"},{"start":44,"end":45,"cssClass":"pl-c1"},{"start":45,"end":51,"cssClass":"pl-s"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":32,"cssClass":"pl-v"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":26,"cssClass":"pl-v"},{"start":27,"end":29,"cssClass":"pl-c1"},{"start":31,"end":32,"cssClass":"pl-c1"},{"start":34,"end":44,"cssClass":"pl-s1"},{"start":44,"end":45,"cssClass":"pl-c1"},{"start":45,"end":51,"cssClass":"pl-s"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":27,"cssClass":"pl-v"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":25,"cssClass":"pl-v"},{"start":26,"end":29,"cssClass":"pl-c1"},{"start":31,"end":41,"cssClass":"pl-s1"},{"start":41,"end":42,"cssClass":"pl-c1"},{"start":42,"end":48,"cssClass":"pl-s"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":27,"cssClass":"pl-v"},{"start":28,"end":31,"cssClass":"pl-c1"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":25,"cssClass":"pl-v"},{"start":26,"end":29,"cssClass":"pl-c1"},{"start":31,"end":41,"cssClass":"pl-s1"},{"start":41,"end":42,"cssClass":"pl-c1"},{"start":42,"end":48,"cssClass":"pl-s"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":27,"cssClass":"pl-v"},{"start":28,"end":31,"cssClass":"pl-c1"}],[{"start":4,"end":6,"cssClass":"pl-s1"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":19,"cssClass":"pl-s1"},{"start":20,"end":25,"cssClass":"pl-v"},{"start":26,"end":28,"cssClass":"pl-c1"},{"start":30,"end":40,"cssClass":"pl-s1"},{"start":40,"end":41,"cssClass":"pl-c1"},{"start":41,"end":50,"cssClass":"pl-s"}],[],[],[{"start":0,"end":21,"cssClass":"pl-s1"},{"start":22,"end":23,"cssClass":"pl-c1"},{"start":24,"end":30,"cssClass":"pl-c1"}],[{"start":0,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":16,"cssClass":"pl-s1"},{"start":17,"end":22,"cssClass":"pl-s1"},{"start":23,"end":33,"cssClass":"pl-s1"},{"start":34,"end":43,"cssClass":"pl-s1"},{"start":44,"end":60,"cssClass":"pl-v"}],[{"start":4,"end":25,"cssClass":"pl-s1"}],[{"start":4,"end":15,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":16,"end":20,"cssClass":"pl-c1"}],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":15,"end":18,"cssClass":"pl-c1"}],[],[{"start":0,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":12,"end":14,"cssClass":"pl-s1"},{"start":15,"end":20,"cssClass":"pl-s1"},{"start":21,"end":31,"cssClass":"pl-s1"},{"start":32,"end":36,"cssClass":"pl-v"},{"start":37,"end":50,"cssClass":"pl-s1"},{"start":50,"end":51,"cssClass":"pl-c1"},{"start":51,"end":62,"cssClass":"pl-s1"}],[],[{"start":0,"end":5,"cssClass":"pl-s1"},{"start":6,"end":13,"cssClass":"pl-en"},{"start":14,"end":23,"cssClass":"pl-s1"},{"start":23,"end":24,"cssClass":"pl-c1"},{"start":24,"end":33,"cssClass":"pl-s1"}],[{"start":14,"end":18,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":19,"end":21,"cssClass":"pl-s1"},{"start":22,"end":27,"cssClass":"pl-s1"},{"start":28,"end":34,"cssClass":"pl-s1"},{"start":35,"end":64,"cssClass":"pl-v"},{"start":65,"end":76,"cssClass":"pl-s1"},{"start":76,"end":77,"cssClass":"pl-c1"},{"start":77,"end":81,"cssClass":"pl-c1"}],[{"start":14,"end":21,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":33,"cssClass":"pl-s"}],[],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":15,"cssClass":"pl-s1"},{"start":16,"end":21,"cssClass":"pl-s1"},{"start":22,"end":31,"cssClass":"pl-s1"},{"start":32,"end":38,"cssClass":"pl-k"},{"start":39,"end":52,"cssClass":"pl-v"}],[{"start":0,"end":2,"cssClass":"pl-s1"},{"start":3,"end":4,"cssClass":"pl-c1"},{"start":5,"end":18,"cssClass":"pl-v"},{"start":19,"end":27,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":28,"end":29,"cssClass":"pl-c1"},{"start":31,"end":51,"cssClass":"pl-s1"},{"start":51,"end":52,"cssClass":"pl-c1"},{"start":52,"end":56,"cssClass":"pl-c1"}],[],[],[{"start":0,"end":5,"cssClass":"pl-s1"},{"start":6,"end":13,"cssClass":"pl-en"},{"start":14,"end":23,"cssClass":"pl-s1"},{"start":23,"end":24,"cssClass":"pl-c1"},{"start":24,"end":30,"cssClass":"pl-s"}],[{"start":14,"end":18,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":19,"end":21,"cssClass":"pl-s1"},{"start":22,"end":27,"cssClass":"pl-s1"},{"start":28,"end":34,"cssClass":"pl-s1"},{"start":35,"end":64,"cssClass":"pl-v"},{"start":65,"end":76,"cssClass":"pl-s1"},{"start":76,"end":77,"cssClass":"pl-c1"},{"start":77,"end":81,"cssClass":"pl-c1"}],[{"start":14,"end":21,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":33,"cssClass":"pl-s"}],[],[],[{"start":0,"end":5,"cssClass":"pl-s1"},{"start":6,"end":9,"cssClass":"pl-en"},{"start":10,"end":18,"cssClass":"pl-s1"}],[{"start":10,"end":19,"cssClass":"pl-s1"},{"start":19,"end":20,"cssClass":"pl-c1"},{"start":20,"end":22,"cssClass":"pl-s1"}],[{"start":10,"end":25,"cssClass":"pl-s1"},{"start":25,"end":26,"cssClass":"pl-c1"},{"start":26,"end":32,"cssClass":"pl-s1"}],[{"start":10,"end":16,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":17,"end":19,"cssClass":"pl-c1"}],[],[{"start":0,"end":5,"cssClass":"pl-s1"},{"start":6,"end":14,"cssClass":"pl-en"},{"start":15,"end":22,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":28,"cssClass":"pl-c"}],[{"start":0,"end":19,"cssClass":"pl-c"}],[{"start":0,"end":13,"cssClass":"pl-c"}],[{"start":0,"end":16,"cssClass":"pl-c"}],[{"start":0,"end":12,"cssClass":"pl-c"}]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":null,"configFilePath":null,"networkDependabotPath":"/Solid32/LeafScan-back/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":false,"repoAlertsPath":"/Solid32/LeafScan-back/security/dependabot","repoSecurityAndAnalysisPath":"/Solid32/LeafScan-back/settings/security_analysis","repoOwnerIsOrg":false,"currentUserCanAdminRepo":true},"displayName":"main.py","displayUrl":"https://github.com/Solid32/LeafScan-back/blob/master/leafscan/main.py?raw=true","headerInfo":{"blobSize":"2.16 KB","deleteInfo":{"deletePath":"https://github.com/Solid32/LeafScan-back/delete/master/leafscan/main.py","deleteTooltip":"Delete this file"},"editInfo":{"editTooltip":"Edit this file"},"ghDesktopPath":"https://desktop.github.com","gitLfsPath":null,"onBranch":true,"shortPath":"533bc41","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2FSolid32%2FLeafScan-back%2Fblob%2Fmaster%2Fleafscan%2Fmain.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"80","truncatedSloc":"65"},"mode":"file"},"image":false,"isCodeownersFile":null,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"Python","large":false,"loggedIn":true,"newDiscussionPath":"/Solid32/LeafScan-back/discussions/new","newIssuePath":"/Solid32/LeafScan-back/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/Solid32/LeafScan-back/blob/master/leafscan/main.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","dismissStackNoticePath":"/settings/dismiss-notice/publish_stack_from_file","releasePath":"/Solid32/LeafScan-back/releases/new?marketplace=true","showPublishActionBanner":false,"showPublishStackBanner":false},"renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"Solid32","repoName":"LeafScan-back","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":null,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timedOut":false,"notAnalyzed":false,"symbols":[{"name":"data_dir","kind":"constant","identStart":218,"identEnd":226,"extentStart":218,"extentEnd":252,"fullyQualifiedName":"data_dir","identUtf16":{"start":{"lineNumber":12,"utf16Col":0},"end":{"lineNumber":12,"utf16Col":8}},"extentUtf16":{"start":{"lineNumber":12,"utf16Col":0},"end":{"lineNumber":12,"utf16Col":34}}},{"name":"train_ds","kind":"constant","identStart":454,"identEnd":462,"extentStart":454,"extentEnd":483,"fullyQualifiedName":"train_ds","identUtf16":{"start":{"lineNumber":21,"utf16Col":0},"end":{"lineNumber":21,"utf16Col":8}},"extentUtf16":{"start":{"lineNumber":21,"utf16Col":0},"end":{"lineNumber":21,"utf16Col":29}}},{"name":"val_ds","kind":"constant","identStart":484,"identEnd":490,"extentStart":484,"extentEnd":509,"fullyQualifiedName":"val_ds","identUtf16":{"start":{"lineNumber":22,"utf16Col":0},"end":{"lineNumber":22,"utf16Col":6}},"extentUtf16":{"start":{"lineNumber":22,"utf16Col":0},"end":{"lineNumber":22,"utf16Col":25}}},{"name":"test_ds","kind":"constant","identStart":510,"identEnd":517,"extentStart":510,"extentEnd":537,"fullyQualifiedName":"test_ds","identUtf16":{"start":{"lineNumber":23,"utf16Col":0},"end":{"lineNumber":23,"utf16Col":7}},"extentUtf16":{"start":{"lineNumber":23,"utf16Col":0},"end":{"lineNumber":23,"utf16Col":27}}},{"name":"mirrored_strategy","kind":"constant","identStart":591,"identEnd":608,"extentStart":591,"extentEnd":643,"fullyQualifiedName":"mirrored_strategy","identUtf16":{"start":{"lineNumber":27,"utf16Col":0},"end":{"lineNumber":27,"utf16Col":17}},"extentUtf16":{"start":{"lineNumber":27,"utf16Col":0},"end":{"lineNumber":27,"utf16Col":52}}},{"name":"initial_learning_rate","kind":"constant","identStart":1339,"identEnd":1360,"extentStart":1339,"extentEnd":1369,"fullyQualifiedName":"initial_learning_rate","identUtf16":{"start":{"lineNumber":46,"utf16Col":0},"end":{"lineNumber":46,"utf16Col":21}},"extentUtf16":{"start":{"lineNumber":46,"utf16Col":0},"end":{"lineNumber":46,"utf16Col":30}}},{"name":"lr_schedule","kind":"constant","identStart":1370,"identEnd":1381,"extentStart":1370,"extentEnd":1501,"fullyQualifiedName":"lr_schedule","identUtf16":{"start":{"lineNumber":47,"utf16Col":0},"end":{"lineNumber":47,"utf16Col":11}},"extentUtf16":{"start":{"lineNumber":47,"utf16Col":0},"end":{"lineNumber":51,"utf16Col":1}}},{"name":"optimizer","kind":"constant","identStart":1502,"identEnd":1511,"extentStart":1502,"extentEnd":1565,"fullyQualifiedName":"optimizer","identUtf16":{"start":{"lineNumber":52,"utf16Col":0},"end":{"lineNumber":52,"utf16Col":9}},"extentUtf16":{"start":{"lineNumber":52,"utf16Col":0},"end":{"lineNumber":52,"utf16Col":63}}},{"name":"es","kind":"constant","identStart":1776,"identEnd":1778,"extentStart":1776,"extentEnd":1833,"fullyQualifiedName":"es","identUtf16":{"start":{"lineNumber":59,"utf16Col":0},"end":{"lineNumber":59,"utf16Col":2}},"extentUtf16":{"start":{"lineNumber":59,"utf16Col":0},"end":{"lineNumber":59,"utf16Col":57}}}]}},"csrf_tokens":{"/Solid32/LeafScan-back/branches":{"post":"0pr-RNvTb-tg4NZWlRAFPoIUr0oQYL_-URpUNUENqC-UXMy99IHgDenpJjPdYT13VORbWM7k9nxXcI62guij_A"}}},"title":"LeafScan-back/main.py at master · Solid32/LeafScan-back","locale":"en"}