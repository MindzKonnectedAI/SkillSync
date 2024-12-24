import axios from "axios";
// import { BACKEND_API_URL } from "./enviroment";
// import { accessToken } from "./enviroment";

export const postRequest = (url, data) => {
//   const auth = localStorage.getItem(accessToken) ? `Bearer ${localStorage.getItem(accessToken)}` : undefined;

  const config = {
    headers: {
    //   Authorization: auth,
      "Content-Type": "application/json;charset=UTF-8",
      "Access-Control-Allow-Origin": "*",
    },
  };
  return axios
    .post(`http://34.170.59.11:3009/api/${url}`, data, config)
    // .post(`https://c88b-2401-4900-884a-9674-1555-5f16-5430-ff9e.ngrok-free.app/api/${url}`, data, config)
    .then((res) => res.data);
};

export const postRequest1 = (url, data) => {
  const auth = localStorage.getItem("auth") ? `Bearer ${localStorage.getItem("auth")}` : undefined;

  const config = {
    headers: {
      Authorization: auth,
      "Content-Type": "multipart/form-data",
      "Access-Control-Allow-Origin": "*",
    },
  };
  return axios
    .post(`http://34.170.59.11:3009/api/${url}`, data, config)
    .then((res) => res.data);
};

// http://127.0.0.1:5000/api/auth/signup
export const getRequest = (url) => {
  const auth = localStorage.getItem("auth") ? `Bearer ${localStorage.getItem("auth")}` : undefined;
  const config = {
    headers: {
      Authorization: auth,
      "Content-Type": "application/json;charset=UTF-8",
      "Access-Control-Allow-Origin": "*",
    },
  };
  return axios.get(`http://34.170.59.11:3009/api/${url}`, config).then((res) => res.data);
  // return axios.get(`https://c88b-2401-4900-884a-9674-1555-5f16-5430-ff9e.ngrok-free.app/api/${url}`, config).then((res) => res.data);
};
