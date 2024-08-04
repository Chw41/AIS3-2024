const CancelToken = axios.CancelToken;
let cancel;

const dqs = (selector, ctx = document) => {
    return ctx.querySelector(selector);
}

HTMLElement.prototype.on = function(event, callback) {
    this.addEventListener(event, callback);
    return this;
}

function updateFile(files, syncWithInput=true) {
    let filename = files[0].name;
    let size = files[0].size;

    // check file size
    if (size >= sizeLimit) {
        ts(".ts.snackbar").snackbar({
            content: "File size is too large!"
        });
        return false;
    }

    if (syncWithInput) {
        dqs("#upload").files = files;
    }

    dqs(".header", dqs("#dragzone")).textContent = filename;
    dqs(".description", dqs("#dragzone")).textContent = `File Size: ${humanFileSize(size, false)}`;
    dqs("#dragzone").dataset.mode = "selected";
}

function reset(ev) {
    if (ev) {
        ev.preventDefault();
        ev.stopPropagation();
    }
    dqs(".header", dqs("#dragzone")).textContent = "Upload";
    dqs(".description", dqs("#dragzone")).innerHTML = "Drag file here to upload, or click here to select file.<br>Max upload size : " + humanFileSize(sizeLimit, false);
    dqs("#dragzone").dataset.mode = "selecting";
    dqs("#upload").value = "";
}

// https://stackoverflow.com/a/14919494
function humanFileSize(bytes, si) {
    var thresh = si ? 1000 : 1024;
    if(Math.abs(bytes) < thresh) {
        return bytes + ' B';
    }
    var units = si
        ? ['kB','MB','GB','TB','PB','EB','ZB','YB']
        : ['KiB','MiB','GiB','TiB','PiB','EiB','ZiB','YiB'];
    var u = -1;
    do {
        bytes /= thresh;
        ++u;
    } while (Math.abs(bytes) >= thresh && u < units.length - 1);
    return bytes.toFixed(1)+' '+units[u];
}

dqs("#upload").on("change", ev => {
    let el = ev.target;
    if (el.files.length) {
        if (el.files.length > 1) {
            ts('.snackbar').snackbar({
                content: "Only 1 file once!"
            });
        } else {
            updateFile(el.files, false);
        }
    } else {
        reset();
    }
});

dqs(".ts.close.button").on("click", ev => {
    if (dqs("#dragzone").dataset.mode == "uploading") {
        if (cancel) {
            cancel();
        }
    }
    reset();
});

dqs("#submitbtn").on("click", ev => {
    ev.stopPropagation();
    ev.preventDefault();

    dqs("#dragzone").dataset.mode = "uploading";

    // clean up styles
    ["preparing", "positive", "negative"].forEach(c => {
        dqs("#progressbar").classList.toggle(c, false);
    });

    dqs("#progressbar .bar").style.width = "0";

    let endpoint = document.form.action;
    axios.post(endpoint, new FormData(document.form), {
        cancelToken: new CancelToken(function (executor) {
            cancel = executor;
        }),
        onUploadProgress: (ev) => {
            percentage = (ev.loaded / ev.total) * 100
            dqs("#progressbar .bar").style.width = percentage + "%";
            if (percentage == 100) {
                dqs("#progressbar").classList.add("preparing");
            }
        }
    }).then(function (res) {
        dqs("#dragzone").dataset.mode = "finished";
        dqs("#progressbar").classList.remove("preparing");
        dqs(".description", dqs("#dragzone")).innerText = res.data.message
    }).catch(function (e) {
        dqs("#dragzone").dataset.mode = "uploadend";
        dqs("#progressbar").classList.remove("preparing");
        dqs("#progressbar").classList.add("negative");
        if (e.response) {
            if (e.response.data) {
                ts(".snackbar").snackbar({
                    content: `Error: ${e.response.data.error}`
                });
            }
        } else if (axios.isCancel(e)) {
            console.log("Upload progress canceled");
            dqs("#progressbar").classList.remove("negative");
            ts(".snackbar").snackbar({
                content: "Upload cancelled"
            });
            dqs("#dragzone").dataset.mode = "selecting";
        } else {
            console.error(e);
        }
    });
});

dqs("#dragzone").on("click", ev => {
    if (dqs("#dragzone").dataset.mode != "uploading") {
        let allowlist = ["button", "a"];
        if (allowlist.indexOf(ev.target.tagName.toLowerCase()) == -1) {
            ev.preventDefault();
            dqs("#upload").click();
        }
    } else {
        ev.preventDefault();
    }
});

dqs("#dragzone").on("drop", ev => {
    ev.stopPropagation();
    ev.preventDefault();

    if (dqs("#dragzone").dataset.mode != "uploading") {
        let files = ev.dataTransfer.files;
        if (files) {
            if (files.length > 1) {
                ts('.snackbar').snackbar({
                    content: "Only 1 file once!"
                });
            } else if (files.length == 1) {
                updateFile(files);
            }
        }
    }
});

["dragenter", "dragover"].forEach(event => {
    dqs("#dragzone").on(event, ev => {
        ev.stopPropagation();
        ev.preventDefault();
    });
});